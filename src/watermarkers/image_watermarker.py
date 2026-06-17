"""
This file defines the abstract base class that Image Watermarkers will inherit from.
"""
from abc import ABC, abstractmethod
import numpy as np

class ImageWatermarker(ABC):
    """
    The abstract base class that future watermarkers will inherit from.
    """
    def __init__(self, configs: dict):
        self.configs = configs
        self.message_length = self.configs["message_length"]

    @abstractmethod
    def encode_image(self, cover: np.ndarray, message: np.ndarray) -> np.ndarray:
        """
        This function should take a cover image and a message and encode the message into the
        cover image, returning the marked image. If there are hyperparameters for encoding the
        message, these should be included in the configs dictionary in the constructor.
        """

    def decode_image(self, stego_image: np.ndarray) -> np.ndarray:
        """
        Given an image, returns the bit array that the image decodes to. I.e., the
        prediction of what the message used to encode the image to begin with actually was.
        """

    @abstractmethod
    def train(self) -> None:
        """
        This function should train anything that will be needed for watermarking later on.
        For example, if the watermarker uses an encoder decoder scheme for its steganogophy, then
        the encoder and decoder would be trained here. Any hyperparameters needed should be
        included in the configs dictionary passed in the constructor.
        """
