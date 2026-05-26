import json
from dataclasses import dataclass

import hydra
import torch
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from omegaconf import DictConfig

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config, separate_dataset_cfg_wrappers
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfgWrapper
    from src.evaluation.evaluation_cfg import EvaluationCfg
    from src.evaluation.metric_computer import MetricComputer
    from src.global_cfg import set_cfg


@dataclass
class RootCfg:
    evaluation: EvaluationCfg
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    seed: int


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="compute_metrics",
)
def evaluate(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg, {list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers},)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    trainer = Trainer(max_epochs=-1, accelerator="gpu")
    computer = MetricComputer(cfg.evaluation)
    data_module = DataModule(cfg.dataset, cfg.data_loader)
    metrics = trainer.test(computer, datamodule=data_module)
    cfg.evaluation.output_metrics_path.parent.mkdir(exist_ok=True, parents=True)
    with cfg.evaluation.output_metrics_path.open("w") as f:
        json.dump(metrics[0], f)


if __name__ == "__main__":
    evaluate()
