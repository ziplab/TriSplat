import copy
import logging
import os
from pathlib import Path

import hydra
import torch
import wandb
from jaxtyping import install_import_hook

logger = logging.getLogger(__name__)
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from src.checkpoint_utils import (
    checkpoint_has_training_state,
    extract_state_dict,
    get_checkpoint_schedule_step,
    load_checkpoint_file,
)
from src.misc.weight_modify import checkpoint_filter_fn_new

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.mesh import get_mesh
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def log_state_dict_load_result(missing_keys, unexpected_keys, source: Path) -> None:
    if missing_keys:
        logger.warning(
            "Missing keys when loading weights from %s: %s",
            source,
            missing_keys,
        )
    if unexpected_keys:
        logger.warning(
            "Unexpected keys when loading weights from %s: %s",
            source,
            unexpected_keys,
        )


def partial_load_gaussian_head(
    state_dict: dict[str, torch.Tensor],
    target_module: torch.nn.Module,
    key_prefix: str = "",
) -> list[str]:
    """Load GS gaussian_head.proj weights into a Triangle model by copying the
    first N dimensions and leaving the extra sigma dimension randomly initialized.

    Returns a list of keys that were partially loaded (and removed from state_dict).
    """
    proj_weight_key = f"{key_prefix}gaussian_head.proj.weight"
    proj_bias_key = f"{key_prefix}gaussian_head.proj.bias"

    partially_loaded = []
    target_params = dict(target_module.named_parameters())

    for key in [proj_weight_key, proj_bias_key]:
        if key not in state_dict:
            continue
        target_key = key[len(key_prefix):] if key_prefix else key
        if target_key not in target_params:
            continue

        src_tensor = state_dict[key]
        tgt_tensor = target_params[target_key]

        if src_tensor.shape == tgt_tensor.shape:
            continue

        src_dim0 = src_tensor.shape[0]
        tgt_dim0 = tgt_tensor.shape[0]
        if src_dim0 >= tgt_dim0:
            continue

        with torch.no_grad():
            tgt_tensor[:src_dim0].copy_(src_tensor)
        logger.info(
            "Partially loaded %s: copied first %d/%d rows from GS checkpoint "
            "(sigma dimension randomly initialized).",
            key, src_dim0, tgt_dim0,
        )
        partially_loaded.append(key)
        del state_dict[key]

    return partially_loaded


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    # load training time evaluation config if needed
    if cfg_dict["mode"] == "train" and cfg_dict["train"]["eval_model_every_n_val"] > 0:
        eval_cfg_dict = copy.deepcopy(cfg_dict)
        for dataset in eval_cfg_dict["dataset"]:
            if dataset == 're10k':
                eval_path = "assets/evaluation_index_re10k_6tx.json"
            elif dataset == 'dl3dv':
                eval_path = "assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json"
            elif dataset == 'scannetpp':
                eval_path = "assets/evaluation_index_scannetpp_iphone_larger.json"
            else:
                raise ValueError(f"unknown dataset={dataset}")
            eval_cfg_dict["dataset"][dataset]["view_sampler"] = {
                "name": "evaluation",
                "index_path": eval_path,
                "num_context_views": eval_cfg_dict["dataset"][dataset]["view_sampler"]["num_context_views"],
            }
        eval_cfg = load_typed_root_config(eval_cfg_dict)
    else:
        eval_cfg = None

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    logger.info(f"Saving outputs to {output_dir}.")

    # Set up logging with wandb.
    callbacks = []

    # Get the Slurm job ID from enviroment variable
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
    tags = cfg_dict.wandb.get("tags", [])
    tags += [f"job_id={slurm_job_id}"] if slurm_job_id != "unknown" else []
    if cfg_dict.wandb.mode != "disabled":
        pl_logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.name})",
            tags=tags,
            log_model=False,
            save_dir=output_dir,
            notes=f"outputs/{output_dir.parent.name}/{output_dir.name}",
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        pl_logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=pl_logger,
        devices="auto",
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        precision=cfg.trainer.precision,
        max_steps=cfg.trainer.max_steps,
        inference_mode=False if (cfg.mode == "test" and (cfg.test.align_pose or cfg.test.post_opt_gs)) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        if "safetensors" in weight_path:
            from safetensors.torch import load_file as torch_load_file
            ckpt_weights = torch_load_file(weight_path, device='cpu')

            ckpt_weights = checkpoint_filter_fn_new(ckpt_weights, encoder, prefix_old='aggregator.', prefix_new='backbone.', gaussians_per_axis=cfg.model.encoder.gaussians_per_axis // cfg.model.encoder.upscale_token_ratio)
            if cfg.model.encoder.use_triangle:
                partial_load_gaussian_head(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
            log_state_dict_load_result(missing_keys, unexpected_keys, Path(weight_path))

        else:
            ckpt_weights = torch.load(weight_path, map_location='cpu')
            if 'state_dict' in ckpt_weights:  # weights trained with our repo
                ckpt_weights = ckpt_weights['state_dict']
                ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
                if cfg.model.encoder.use_triangle:
                    partial_load_gaussian_head(ckpt_weights, encoder)
                missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
                log_state_dict_load_result(missing_keys, unexpected_keys, Path(weight_path))

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        get_mesh(cfg.mesh),
        eval_data_cfg=(
            None if eval_cfg is None else eval_cfg.dataset
        ),
        gaussian_downsample_ratio=cfg.model.encoder.gaussian_downsample_ratio,
        gaussians_per_axis=cfg.model.encoder.gaussians_per_axis,
    )
    model_wrapper.gradient_clip_val = cfg.trainer.gradient_clip_val
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    fit_checkpoint_path = checkpoint_path
    test_checkpoint_path = checkpoint_path
    if cfg.mode == "train" and checkpoint_path is not None:
        checkpoint = load_checkpoint_file(checkpoint_path)
        if checkpoint_has_training_state(checkpoint):
            ckpt_state_dict = extract_state_dict(checkpoint)
            try:
                missing_keys, unexpected_keys = model_wrapper.load_state_dict(
                    ckpt_state_dict,
                    strict=False,
                )
                if missing_keys or unexpected_keys:
                    logger.info(
                        "Checkpoint %s contains trainer state, but the current model structure changed. "
                        "Loading weights only and starting a new training run.",
                        checkpoint_path,
                    )
                    log_state_dict_load_result(missing_keys, unexpected_keys, checkpoint_path)
                    fit_checkpoint_path = None
                else:
                    logger.info("Resuming training state from %s.", checkpoint_path)
            except RuntimeError as err:
                logger.info(
                    "Checkpoint %s is not directly compatible with the current model (%s). "
                    "Loading weights only and starting a new training run.",
                    checkpoint_path,
                    err,
                )
                ckpt_state_dict = extract_state_dict(checkpoint)
                if cfg.model.encoder.use_triangle:
                    partial_load_gaussian_head(
                        ckpt_state_dict, model_wrapper.encoder, key_prefix="encoder."
                    )
                missing_keys, unexpected_keys = model_wrapper.load_state_dict(
                    ckpt_state_dict,
                    strict=False,
                )
                log_state_dict_load_result(missing_keys, unexpected_keys, checkpoint_path)
                fit_checkpoint_path = None
        else:
            logger.info(
                "Checkpoint %s does not contain trainer state. Loading model weights only and starting a new training run.",
                checkpoint_path,
            )
            ckpt_state_dict = extract_state_dict(checkpoint)
            if cfg.model.encoder.use_triangle:
                partial_load_gaussian_head(
                    ckpt_state_dict, model_wrapper.encoder, key_prefix="encoder."
                )
            missing_keys, unexpected_keys = model_wrapper.load_state_dict(
                ckpt_state_dict,
                strict=False,
            )
            log_state_dict_load_result(missing_keys, unexpected_keys, checkpoint_path)
            fit_checkpoint_path = None
    elif cfg.mode == "test" and checkpoint_path is not None:
        checkpoint = load_checkpoint_file(checkpoint_path)
        if not checkpoint_has_training_state(checkpoint):
            logger.info(
                "Checkpoint %s does not contain trainer state. Loading model weights only for evaluation.",
                checkpoint_path,
            )
            ckpt_state_dict = extract_state_dict(checkpoint)
            if cfg.model.encoder.use_triangle:
                partial_load_gaussian_head(
                    ckpt_state_dict, model_wrapper.encoder, key_prefix="encoder."
                )
            missing_keys, unexpected_keys = model_wrapper.load_state_dict(
                ckpt_state_dict,
                strict=False,
            )
            log_state_dict_load_result(missing_keys, unexpected_keys, checkpoint_path)
            model_wrapper.restored_global_step = get_checkpoint_schedule_step(
                checkpoint,
                checkpoint_path,
            )
            test_checkpoint_path = None

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=fit_checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=test_checkpoint_path,
        )


if __name__ == "__main__":
    train()
