import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def se3_inverse(T):
    """
    Computes the inverse of a batch of SE(3) matrices.
    """

    if torch.is_tensor(T):
        R = T[..., :3, :3]
        t = T[..., :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.cat([
            torch.cat([R_inv, t_inv], dim=-1),
            torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(*T.shape[:-2], 1, 1)
        ], dim=-2)
    else:
        R = T[..., :3, :3]
        t = T[..., :3, 3, np.newaxis]

        R_inv = np.swapaxes(R, -2, -1)
        t_inv = -R_inv @ t

        bottom_row = np.zeros((*T.shape[:-2], 1, 4), dtype=T.dtype)
        bottom_row[..., :, 3] = 1

        top_part = np.concatenate([R_inv, t_inv], axis=-1)
        T_inv = np.concatenate([top_part, bottom_row], axis=-2)

    return T_inv


class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:
            R_err: rotation angular error
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()  # [0, 3.14]

    def forward(self, pred_pose, gt_pose, scale=None):
        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        if scale is not None:
            pred_pose_align[..., :3, 3] *= scale.view(B, 1, 1)

        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)

        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)

        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)

        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]

        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)

        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3),
            R_gt.reshape(-1, 3, 3)
        )

        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)
