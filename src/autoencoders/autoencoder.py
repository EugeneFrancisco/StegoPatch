"""
This file defines the base class for autoencoders to inherit from.
"""
from abc import ABC, abstractmethod
import numpy as np

class AutoEncoder(ABC):
    """
    This abstract base class should be inherited by autoencoders.
    """
    @abstractmethod
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Returns the latent representation of the image.
        """

    @abstractmethod
    def decode(self, latent_variable: np.ndarray) -> np.ndarray:
        """
        Returns the image representation of the latent variable.
        """
