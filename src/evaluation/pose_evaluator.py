import logging

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import pose_auc, update_pose, get_pnp_pose

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from lightning import LightningModule
from tabulate import tabulate

from ..loss.loss_ssim import ssim
from ..misc.utils import get_overlap_tag
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_pose_error

logger = logging.getLogger(__name__)


class PoseEvaluator(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg, encoder, decoder, losses) -> None:
        super().__init__()
        self.cfg = cfg

        # our model
        self.encoder = encoder.to(self.device)
        self.decoder = decoder
        self.losses = nn.ModuleList(losses)

        self.data_shim = get_data_shim(self.encoder)

        self.benchmarker = Benchmarker()
        self.time_skip_first_n_steps = 10
        self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

        self.all_mertrics = {}
        self.all_mertrics_sub = {}

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        # set to eval
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            logger.info(f"Test step {batch_idx:0>6}.")

        # # get overlap.
        # overlap = batch["context"]["overlap"][0, 0]
        # overlap_tag = get_overlap_tag(overlap)
        # if overlap_tag == "ignore":
        #     return
        overlap_tag = None

        if batch_idx < self.time_skip_first_n_steps:
            self.time_skip_steps_dict["encoder"] += 1
            self.time_skip_steps_dict["decoder"] += v

        visualization_dump = {}
        with self.benchmarker.time("encoder"):
            _ = self.encoder(
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump,
        )

        extrinsic = visualization_dump['pred_camera_poses']

        # eval pose
        error_ts, error_Rs, error_poses = [], [], []
        # Method 1: only eval performance compared with the first view
        for i in range(1, v):
            gt_pose = batch["context"]["extrinsics"][0, i]
            eval_pose = extrinsic[0, i]
            error_t, error_t_scale, error_R = compute_pose_error(gt_pose, eval_pose)
            error_pose = torch.max(error_t, error_R)  # find the max error
            error_ts.append(error_t)
            error_Rs.append(error_R)
            error_poses.append(error_pose)
        error_t = torch.mean(torch.stack(error_ts))
        error_R = torch.mean(torch.stack(error_Rs))
        error_pose = torch.mean(torch.stack(error_poses))

        all_metrics = {
            "e_t_ours": error_t,
            "e_R_ours": error_R,
            "e_pose_ours": error_pose,
        }

        self.log_dict(all_metrics)
        self.print_preview_metrics(all_metrics, overlap_tag)

        return 0

    def calculate_auc(self, tot_e_pose, method_name, overlap_tag):
        thresholds = [5, 10, 20]
        auc = pose_auc(tot_e_pose, thresholds)
        print(f"Pose AUC {method_name} {overlap_tag}: ")
        print(auc)
        return auc

    def on_test_end(self) -> None:
        # eval pose
        for method in self.cfg.methods:
            tot_e_pose = np.array(self.all_mertrics[f"e_pose_{method.key}"])
            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            print(f"Pose AUC {method.key}: ")
            print(auc)

            for overlap_tag in self.all_mertrics_sub.keys():
                tot_e_pose = np.array(self.all_mertrics_sub[overlap_tag][f"e_pose_{method.key}"])
                tot_e_pose = np.array(tot_e_pose)
                thresholds = [5, 10, 20]
                auc = pose_auc(tot_e_pose, thresholds)
                print(f"Pose AUC {method.key} {overlap_tag}: ")
                print(auc)

        # save all metrics
        np.save("all_metrics.npy", self.all_mertrics)
        np.save("all_metrics_sub.npy", self.all_mertrics_sub)

        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(self.time_skip_steps_dict[tag]):]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
        self.benchmarker.summarize_memory()

    def print_preview_metrics(self, metrics: dict[str, float], overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1

            self.all_mertrics = {k: [v.cpu().item()] for k, v in metrics.items()}
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

            for k, v in metrics.items():
                self.all_mertrics[k].append(v.cpu().item())

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
                self.all_mertrics_sub = {overlap_tag: {k: [v.cpu().item()] for k, v in metrics.items()}}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
                self.all_mertrics_sub[overlap_tag] = {k: [v.cpu().item()] for k, v in metrics.items()}
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

                for k, v in metrics.items():
                    self.all_mertrics_sub[overlap_tag][k].append(v.cpu().item())

        def print_metrics(runing_metric):
            table = []
            for method in self.cfg.methods:
                row = [
                    f"{runing_metric[f'{metric}_{method.key}']:.3f}"
                    for metric in ("e_t", "e_R", "e_pose")
                ]
                table.append((method.key, *row))

            table = tabulate(table, ["Method", "e_t", "e_R", "e_pose"])
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v)
