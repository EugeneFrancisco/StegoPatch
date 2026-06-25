"""
This file defines a noising class for use with the stegopatch watermarker. It is identical to the
RoSteALSNoiser except that it adds a cropping noise: when cropping is sampled, the whole batch is
cropped to a single random rectangular window whose height and width are sampled independently and
uniformly, each at least crop_size and at most the corresponding image dimension.
"""
from typing import Union, Callable
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.noisers.rosteals_noiser import RoSteALSNoiser


class StegoPatchNoiser(RoSteALSNoiser):
    CROP = 3
    ROTATE = 4
    _NAME_TO_INT = {**RoSteALSNoiser._NAME_TO_INT, "crop": CROP, "rotate": ROTATE}

    def __init__(self, configs: dict):
        super().__init__(configs)
        c = self.configs

        # Require every branch probability to be set explicitly (fail loudly otherwise).
        self.set_probabilities(
            p_identity=c["p_identity"],
            p_differentiable=c["p_differentiable"],
            p_imagenet=c["p_imagenet"],
            p_crop=c["p_crop"],
            p_rotate=c["p_rotate"],
        )

        # The (square) side length of the random crop window.
        self.crop_size = int(c["crop_size"])

        # Inclusive bounds (in degrees) for the uniformly sampled rotation angle.
        self.rotation_lower_bound = float(c["rotation_lower_bound"])
        self.rotation_upper_bound = float(c["rotation_upper_bound"])

    # -- probability control -------------------------------------------------
    def set_probabilities(
        self,
        p_identity: float,
        p_differentiable: float,
        p_imagenet: float,
        p_crop: float,
        p_rotate: float,
    ) -> None:
        """Overwrite the branch sampling probabilities. They must sum to 1."""
        assert abs(p_identity + p_differentiable + p_imagenet + p_crop + p_rotate - 1.0) < 1e-6
        self.p_identity = p_identity
        self.p_diff = p_differentiable
        self.p_imagenet = p_imagenet
        self.p_crop = p_crop
        self.p_rotate = p_rotate

    # -- type normalisation --------------------------------------------------
    def _normalize_type(self, noise_type: Union[str, int]) -> int:
        if noise_type == self.CROP or (
            isinstance(noise_type, str) and noise_type.lower() == "crop"
        ):
            return self.CROP
        if noise_type == self.ROTATE or (
            isinstance(noise_type, str) and noise_type.lower() == "rotate"
        ):
            return self.ROTATE
        return super()._normalize_type(noise_type)

    # -- dispatch ------------------------------------------------------------
    def get_noise_function(
        self, noise_type: Union[str, int]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        t = self._normalize_type(noise_type)
        if t == self.CROP:
            return self._crop_noise
        if t == self.ROTATE:
            return self._rotate_noise
        return super().get_noise_function(noise_type)

    def sample_noise_type(self) -> int:
        types = [self.DIFFERENTIABLE, self.IMAGENET, self.IDENTITY, self.CROP, self.ROTATE]
        probs = [self.p_diff, self.p_imagenet, self.p_identity, self.p_crop, self.p_rotate]
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

    # -- rotate --------------------------------------------------------------
    @staticmethod
    def _inscribed_hw(h: int, w: int, angle_deg: float) -> tuple:
        """Return the (height, width) of the largest axis-aligned rectangle that fits entirely
        inside an h x w image rotated by angle_deg degrees, i.e. a rectangle containing only real
        image pixels and no black corners. Standard largest-inscribed-rectangle solution."""
        if h <= 0 or w <= 0:
            return h, w
        angle = math.radians(angle_deg)
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr = (w * cos_a - h * sin_a) / cos_2a
            hr = (h * cos_a - w * sin_a) / cos_2a

        # Inset by 1px so the outermost ring (faintly darkened by bilinear sampling at the
        # rotated frame edge) is excluded, leaving only clean interior pixels.
        return max(1, int(math.floor(hr)) - 1), max(1, int(math.floor(wr)) - 1)

    def _rotate_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the whole (B, C, H, W) batch by a single angle sampled uniformly from
        [rotation_lower_bound, rotation_upper_bound] degrees (counter-clockwise), then centre-crop
        to the largest rectangle of real image pixels. This "zooms in" so no black corners remain;
        the spatial dimensions shrink as the rotation angle grows. Bilinear interpolation and the
        slice-based crop are both differentiable, so gradients flow back to the input pixels."""
        angle = float(np.random.uniform(self.rotation_lower_bound, self.rotation_upper_bound))
        rotated = TF.rotate(
            x,
            angle=angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=False,
        )
        _, _, h, w = x.shape
        hr, wr = self._inscribed_hw(h, w, angle)
        return TF.center_crop(rotated, [hr, wr])

    # -- numpy entry point ---------------------------------------------------
    def apply_noise_np(self, image: np.ndarray, noise_type: Union[str, int]) -> np.ndarray:
        t = self._normalize_type(noise_type)
        if t == self.CROP:
            h, w = image.shape[:2]
            ch = int(np.random.randint(self.crop_size, h + 1))
            cw = int(np.random.randint(self.crop_size, w + 1))
            top = int(np.random.randint(0, h - ch + 1))
            left = int(np.random.randint(0, w - cw + 1))
            return image[top:top + ch, left:left + cw].copy()
        if t == self.ROTATE:
            was_float = np.issubdtype(image.dtype, np.floating)
            x = image.astype(np.float32) if was_float else image.astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                x = self._rotate_noise(x)
            out = x.squeeze(0).permute(1, 2, 0).numpy()
            return out if was_float else (out * 255).astype(np.uint8)
        return super().apply_noise_np(image, noise_type)
