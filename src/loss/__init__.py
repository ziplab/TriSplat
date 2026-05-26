from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_intrinsic import LossIntrinsic, LossIntrinsicCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_normal_consistency import LossNormalConsistency, LossNormalConsistencyCfgWrapper
from .loss_normal_teacher import LossNormalTeacher, LossNormalTeacherCfgWrapper
from .loss_opacity import LossOpacity, LossOpacityCfgWrapper
from .loss_perceptual import LossPerceptual, LossPerceptualCfgWrapper
from .loss_pose_cfg import LossPose, LossPoseCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossIntrinsicCfgWrapper: LossIntrinsic,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossNormalConsistencyCfgWrapper: LossNormalConsistency,
    LossNormalTeacherCfgWrapper: LossNormalTeacher,
    LossOpacityCfgWrapper: LossOpacity,
    LossPerceptualCfgWrapper: LossPerceptual,
    LossPoseCfgWrapper: LossPose,
}

LossCfgWrapper = (
    LossDepthCfgWrapper
    | LossIntrinsicCfgWrapper
    | LossLpipsCfgWrapper
    | LossMseCfgWrapper
    | LossNormalConsistencyCfgWrapper
    | LossNormalTeacherCfgWrapper
    | LossOpacityCfgWrapper
    | LossPerceptualCfgWrapper
    | LossPoseCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
