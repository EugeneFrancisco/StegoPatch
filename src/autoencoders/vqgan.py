"""
This file defines a subclass of the AutoEncoder class which implements an autoencoder
using a pretrained VQGAN model.

The heavy lifting (model definition + weights) is delegated to the ``diffusers``
library's ``VQModel``, which exposes any VQGAN published on the Hugging Face Hub.
Images are assumed to be ``np.ndarray`` of shape ``(H, W, C)`` with float values in
``[0, 1]`` (the standard convention), and latents are returned/accepted as
``np.ndarray`` of shape ``(latent_channels, H // shrink_factor, W // shrink_factor)``.
"""
import numpy as np
import torch
from diffusers import VQModel

from src.autoencoders.autoencoder import AutoEncoder


class VQGAN(AutoEncoder):
    """
    AutoEncoder backed by a pretrained VQGAN (``diffusers.VQModel``).

    Expected ``configs`` keys:
        pretrained_model: Hugging Face Hub id or local path of the VQGAN.
        shrink_factor:    Spatial downsampling factor of the encoder (e.g. 4 or 8).
        latent_channels:  Number of channels in the latent representation.
        subfolder:        (optional) Subfolder within the repo holding the model.
        device:           (optional) Torch device string; defaults to cuda/mps/cpu.
    """

    def __init__(self, configs: dict):
        self.configs = configs
        self.shrink_factor: int = configs["shrink_factor"]
        self.latent_channels: int = configs["latent_channels"]
        self.device = torch.device(configs.get("device", self._default_device()))

        self.model = VQModel.from_pretrained(
            configs["pretrained_model"], subfolder=configs.get("subfolder")
        )
        self.model.eval().to(self.device)
        self._validate_config()

    @staticmethod
    def _default_device() -> str:
        """Picks the best available torch device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _validate_config(self) -> None:
        """
        Sanity-checks that the loaded model matches the advertised ``configs`` so a
        mismatch fails loudly here rather than as a confusing shape error later.
        """
        model_latent_channels = self.model.config.latent_channels
        # A VQGAN halves the spatial size once per downsampling block, and the first
        # block out channel does not downsample, hence the ``- 1``.
        model_shrink_factor = 2 ** (len(self.model.config.block_out_channels) - 1)

        if model_latent_channels != self.latent_channels:
            raise ValueError(
                f"configs['latent_channels']={self.latent_channels} does not match the "
                f"loaded model's {model_latent_channels}."
            )
        if model_shrink_factor != self.shrink_factor:
            raise ValueError(
                f"configs['shrink_factor']={self.shrink_factor} does not match the "
                f"loaded model's {model_shrink_factor}."
            )

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Returns the (continuous, pre-quantization) latent representation of the image.

        Quantization is deferred to :meth:`decode` so that downstream watermarking can
        operate on the continuous latent space, following Bui et al.
        """
        tensor = self._image_to_tensor(image)
        with torch.no_grad():
            latents = self.model.encode(tensor).latents
        return latents.squeeze(0).cpu().numpy()

    def decode(self, latent_variable: np.ndarray) -> np.ndarray:
        """
        Returns the image reconstructed from a latent produced by :meth:`encode`.

        The latent is vector-quantized to the nearest codebook entries before being
        decoded back into pixel space.
        """
        tensor = torch.from_numpy(latent_variable).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            image = self.model.decode(tensor).sample
        return self._tensor_to_image(image)

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Converts an ``(H, W, C)`` image in ``[0, 1]`` to a ``(1, C, H, W)`` tensor in ``[-1, 1]``."""
        tensor = torch.from_numpy(image).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(self.device)

    @staticmethod
    def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """Converts a ``(1, C, H, W)`` tensor in ``[-1, 1]`` back to an ``(H, W, C)`` image in ``[0, 1]``."""
        tensor = (tensor.squeeze(0) / 2.0 + 0.5).clamp(0.0, 1.0)
        return tensor.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
