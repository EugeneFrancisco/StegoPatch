"""
This file defines an inheritor of the noiser to mimic the noising scheme from Bui et al.
"""
from typing import Union, Callable
from src.noisers.imagenet_corruptions import corrupt, DEFAULT_CORRUPTION_IDS
from src.noisers.noiser import Noiser
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

class RoSteALSNoiser(Noiser):
    IDENTITY = 0
    DIFFERENTIABLE = 1
    IMAGENET = 2
    _NAME_TO_INT = {
        "identity": IDENTITY,
        "differentiable": DIFFERENTIABLE,
        "imagenet": IMAGENET,
    }
    # RoSteALS training corruption ids, minus the ImageMagick-backed ones
    # (glass/motion/zoom blur, snow) and frost (needs bundled assets).
    _DEFAULT_CORRUPTIONS = DEFAULT_CORRUPTION_IDS
    # standard luma weights, used for the saturation blend
    _LUMA = (0.299, 0.587, 0.114)

    def __init__(self, configs: dict):
        super().__init__(configs)
        c = self.configs

        # branch probabilities
        self.p_diff = c.get("p_differentiable", 0.45)
        self.p_imagenet = c.get("p_imagenet", 0.45)
        self.p_identity = c.get("p_identity", 0.10)

        # differentiable-chain params
        self.blur_kernel_size = int(c.get("blur_kernel_size", 7))
        self.blur_sigma_range = c.get("blur_sigma_range", [0.1, 1.5])
        self.noise_std = c.get("noise_std", 0.04)
        self.contrast_range = c.get("contrast_range", [0.8, 1.2])
        self.brightness_range = c.get("brightness_range", [-0.1, 0.1])
        self.saturation_range = c.get("saturation_range", [0.0, 0.5])

        # imagenet-c params
        self.corruption_ids = list(c.get("imagenet_corruptions", self._DEFAULT_CORRUPTIONS))
        self.severity_range = c.get("imagenet_severity_range", [1, 5])

    # -- type normalisation --------------------------------------------------
    def _normalize_type(self, noise_type: Union[str, int]) -> int:
        if isinstance(noise_type, str):
            key = noise_type.lower()
            if key not in self._NAME_TO_INT:
                raise ValueError(f"Unknown noise type: {noise_type!r}")
            return self._NAME_TO_INT[key]
        if noise_type in (self.IDENTITY, self.DIFFERENTIABLE, self.IMAGENET):
            return int(noise_type)
        raise ValueError(f"Unknown noise type: {noise_type!r}")

    # -- dispatch ------------------------------------------------------------
    def get_noise_function(
        self, noise_type: Union[str, int]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        t = self._normalize_type(noise_type)
        if t == self.IDENTITY:
            return lambda x: x
        if t == self.DIFFERENTIABLE:
            return self._differentiable_noise
        return self._imagenet_noise  # IMAGENET

    def sample_noise_type(self) -> int:
        types = [self.DIFFERENTIABLE, self.IMAGENET, self.IDENTITY]
        probs = [self.p_diff, self.p_imagenet, self.p_identity]
        return int(np.random.choice(types, p=probs))

    # -- differentiable chain ------------------------------------------------
    def _gaussian_kernel(self, ksize: int, sigma: float, device) -> torch.Tensor:
        ax = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2.0
        g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        return torch.outer(g, g)  # (k, k)

    def _differentiable_noise(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, _, _ = x.shape
        device = x.device

        # blur: one random kernel for the batch
        sigma = float(np.random.uniform(*self.blur_sigma_range))
        k = self._gaussian_kernel(self.blur_kernel_size, sigma, device)
        weight = k.view(1, 1, *k.shape).expand(ch, 1, *k.shape)
        x = F.conv2d(x, weight, padding=self.blur_kernel_size // 2, groups=ch)  # pylint: disable=not-callable

        # additive gaussian noise (per-pixel)
        x = x + torch.randn_like(x) * self.noise_std

        # contrast: per-image scale
        contrast = torch.empty(b, 1, 1, 1, device=device).uniform_(*self.contrast_range)
        x = x * contrast

        # brightness + hue: per-image, per-channel offset
        bright = torch.empty(b, ch, 1, 1, device=device).uniform_(*self.brightness_range)
        x = x + bright

        # saturation: per-image blend toward luminance
        sat = torch.empty(b, 1, 1, 1, device=device).uniform_(*self.saturation_range)
        luma = torch.tensor(self._LUMA, device=device).view(1, ch, 1, 1)
        lum = (x * luma).sum(dim=1, keepdim=True)
        x = (1.0 - sat) * x + sat * lum

        return x.clamp(0.0, 1.0)

    # -- imagenet-c (straight-through) ---------------------------------------
    def _corrupt_single(
            self,
            img_uint8: np.ndarray,
            corruption_id: int,
            severity: int
        ) -> np.ndarray:
        """Corrupt one HxWxC uint8 RGB image; resize to 224 like RoSteALS then back."""
        h, w = img_uint8.shape[:2]
        pil = Image.fromarray(img_uint8).resize((224, 224), Image.Resampling.BILINEAR)
        out = corrupt(np.asarray(pil), severity=severity, corruption_number=corruption_id)
        out = Image.fromarray(out).resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(out)

    def _imagenet_noise(self, x: torch.Tensor) -> torch.Tensor:
        # build corrupted batch on cpu/numpy (detached), one corruption per image
        x_np = (x.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        x_np = x_np.transpose(0, 2, 3, 1)  # (B, H, W, C)

        lo, hi = self.severity_range
        out = []
        for img in x_np:
            cid = int(np.random.choice(self.corruption_ids))
            sev = int(np.random.randint(lo, hi + 1))
            out.append(self._corrupt_single(img, cid, sev))
        out = np.stack(out).astype(np.float32) / 255.0
        out = out.transpose(0, 3, 1, 2)  # (B, C, H, W)

        corrupted = torch.from_numpy(out).to(x.device)
        # straight-through: forward = corrupted, backward = identity
        return x + (corrupted - x.detach())

    # -- numpy entry point ---------------------------------------------------
    def apply_noise_np(self, image: np.ndarray, noise_type: Union[str, int]) -> np.ndarray:
        """Apply one noise type to a single HxWxC image (uint8 or float in [0,1])."""
        t = self._normalize_type(noise_type)
        if t == self.IDENTITY:
            return image.copy()

        was_float = np.issubdtype(image.dtype, np.floating)
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) if was_float else image

        if t == self.IMAGENET:
            lo, hi = self.severity_range
            cid = int(np.random.choice(self.corruption_ids))
            sev = int(np.random.randint(lo, hi + 1))
            out = self._corrupt_single(img_uint8, cid, sev)
        else:  # DIFFERENTIABLE: reuse the tensor path
            x = torch.from_numpy(img_uint8.astype(np.float32) / 255.0)
            x = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                x = self._differentiable_noise(x)
            out = (x.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        return out.astype(np.float32) / 255.0 if was_float else out
