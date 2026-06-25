"""
This file defines a noising class for use with the stegopatch watermarker. It is identical to the
RoSteALSNoiser except that it adds a cropping noise: when cropping is sampled, the whole batch is
cropped to a single random rectangular window whose height and width are sampled independently and
uniformly, each at least crop_size and at most the corresponding image dimension.
"""
from typing import Union, Callable
import numpy as np
import torch

from src.noisers.rosteals_noiser import RoSteALSNoiser


class StegoPatchNoiser(RoSteALSNoiser):
    CROP = 3
    _NAME_TO_INT = {**RoSteALSNoiser._NAME_TO_INT, "crop": CROP}

    def __init__(self, configs: dict):
        super().__init__(configs)
        c = self.configs

        # Require every branch probability to be set explicitly (fail loudly otherwise).
        self.set_probabilities(
            p_identity=c["p_identity"],
            p_differentiable=c["p_differentiable"],
            p_imagenet=c["p_imagenet"],
            p_crop=c["p_crop"],
        )

        # The (square) side length of the random crop window.
        self.crop_size = int(c["crop_size"])

    # -- probability control -------------------------------------------------
    def set_probabilities(
        self,
        p_identity: float,
        p_differentiable: float,
        p_imagenet: float,
        p_crop: float,
    ) -> None:
        """Overwrite the branch sampling probabilities. They must sum to 1."""
        assert abs(p_identity + p_differentiable + p_imagenet + p_crop - 1.0) < 1e-6
        self.p_identity = p_identity
        self.p_diff = p_differentiable
        self.p_imagenet = p_imagenet
        self.p_crop = p_crop

    # -- type normalisation --------------------------------------------------
    def _normalize_type(self, noise_type: Union[str, int]) -> int:
        if noise_type == self.CROP or (
            isinstance(noise_type, str) and noise_type.lower() == "crop"
        ):
            return self.CROP
        return super()._normalize_type(noise_type)

    # -- dispatch ------------------------------------------------------------
    def get_noise_function(
        self, noise_type: Union[str, int]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if self._normalize_type(noise_type) == self.CROP:
            return self._crop_noise
        return super().get_noise_function(noise_type)

    def sample_noise_type(self) -> int:
        types = [self.DIFFERENTIABLE, self.IMAGENET, self.IDENTITY, self.CROP]
        probs = [self.p_diff, self.p_imagenet, self.p_identity, self.p_crop]
        return int(np.random.choice(types, p=probs))

    # -- crop ----------------------------------------------------------------
    def _crop_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Crop the whole (B, C, H, W) batch to a single random rectangular window. The crop
        height and width are sampled independently and uniformly, each at least crop_size and at
        most the corresponding image dimension.
        Slicing is differentiable, so gradients flow through to the kept region."""
        _, _, h, w = x.shape
        ch = int(np.random.randint(self.crop_size, h + 1))
        cw = int(np.random.randint(self.crop_size, w + 1))
        top = int(np.random.randint(0, h - ch + 1))
        left = int(np.random.randint(0, w - cw + 1))
        return x[:, :, top:top + ch, left:left + cw]

    # -- numpy entry point ---------------------------------------------------
    def apply_noise_np(self, image: np.ndarray, noise_type: Union[str, int]) -> np.ndarray:
        if self._normalize_type(noise_type) == self.CROP:
            h, w = image.shape[:2]
            ch = int(np.random.randint(self.crop_size, h + 1))
            cw = int(np.random.randint(self.crop_size, w + 1))
            top = int(np.random.randint(0, h - ch + 1))
            left = int(np.random.randint(0, w - cw + 1))
            return image[top:top + ch, left:left + cw].copy()
        return super().apply_noise_np(image, noise_type)
