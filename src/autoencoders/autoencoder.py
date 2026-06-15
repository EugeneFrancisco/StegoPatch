"""
This file defines the base class for autoencoders to inherit from.

Autoencoders are an internal, torch-native interface: they operate on batched
(B, C, H, W) tensors and are differentiable, so gradients can flow through them
during training. Conversion to/from numpy happens at the watermarker's edge.
"""
from abc import ABC, abstractmethod
import torch

class AutoEncoder(ABC):
    """
    This abstract base class should be inherited by autoencoders.
    """
    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent representation of a batch of images.
        """

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Returns the image representation of a batch of latent variables.
        """

    @classmethod
    def get_latent_dim(cls, h_image: int, w_image: int) -> tuple:
        """
        Returns latent space dimension.
        """
