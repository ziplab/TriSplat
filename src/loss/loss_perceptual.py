import os
from dataclasses import dataclass

import scipy.io
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from lightning.pytorch.utilities import rank_zero_only
from torchvision.models import vgg19
from torch import Tensor
from jaxtyping import Float

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Primitives
from .loss import Loss


@dataclass
class LossPerceptualCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossPerceptualCfgWrapper:
    perceptual: LossPerceptualCfg


# the perception loss code is modified from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
# and some parts are based on https://github.com/arthurhero/Long-LRM/blob/main/model/loss.py
class LossPerceptual(Loss[LossPerceptualCfg, LossPerceptualCfgWrapper]):
    def __init__(self, cfg: LossPerceptualCfgWrapper):
        super().__init__(cfg)
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()

        convert_to_buffer(self.vgg, persistent=False)
        convert_to_buffer(self.blocks, persistent=False)

    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)

        return model.eval()

    @rank_zero_only
    def _maybe_download_weights(self, weight_file):
        """Download weights if needed - runs only on rank 0 or non-distributed."""
        if not weight_file.exists():
            os.system(
                f'wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}')

    def _load_weights(self):
        """Load pre-trained VGG weights. """
        weight_file = Path("./pretrained_weights/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)

        self._maybe_download_weights(weight_file)

        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]

        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Set weights
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)

                # Set biases
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)

    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()

        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i + 1]]))
            self.blocks.append(block.eval())

        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        # VGG mean values for ImageNet
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean

    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Primitives,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        if use_context:
            target_img = batch["context"]["image"]
        else:
            target_img = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=target_img.device)

        target_img = rearrange(target_img, "b v c h w -> (b v) c h w")
        pred_img = rearrange(prediction.color, "b v c h w -> (b v) c h w")

        # Preprocess images
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)

        # Extract features
        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)

        # Pixel-level error
        e0 = self._compute_error(target_img_p, pred_img_p)

        # Feature-level errors with scaling factors
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5

        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0

        return self.cfg.weight * total_loss
