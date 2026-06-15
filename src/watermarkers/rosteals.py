"""
This file defines an image watermarker using RoSteALS method from Bui et al.
"""
import numpy as np
import torch
import torch.nn as nn
from src.watermarkers.image_watermarker import ImageWatermarker
from src.autoencoders.vqgan import VQGAN

class Reshape(nn.Module):
    """
    Wrapper class for reshaping.
    """
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """
        Reshape x.
        """
        return x.reshape(x.shape[0], *self.shape)

class RoSteALS(ImageWatermarker):
    """
    An implementation of the RoSteALS watermarker from Bui et al.
    """
    def __init__(self, configs: dict):
        """
        Configs should have the following keys:
            device (str) the device that we will use (e.g., mps)
            autoencoder_type (str) the type of autoencoder that is being used.
            message_length (int) the length of the message we are trying to send.
            c_image (int) num channels in the image.
            h_image (int) height of the image.
            w_image (int) width of the image.
            h_little (int) height of the intermediate for the message encoder.
            w_little (int) width of the intermediate for the message encoder.
            c_little (int) num channels of the intermediate for the message encoder.
        """
        super().__init__(configs)

        self.device = configs["device"]

        # The depth of the images.
        self.c_image: int = int(self.configs["c_image"])

        # The height and width of images that will be watermarked.
        self.h_image: int = int(self.configs["h_image"])
        self.w_image: int = int(self.configs["w_image"])

        self.autoencoder_type = self.configs["autoencoder_type"]
        if self.autoencoder_type == "VQGAN":
            # The image autoencoder that can encode images to a latent space, and decode
            # latent variables into images.
            self.image_autoencoder = VQGAN(device = self.device)
            self.latent_shape: tuple[int] = VQGAN.get_latent_dim(self.h_image, self.w_image)

        assert len(self.latent_shape) == 3

        # The number of channels in the latent representation.
        self.c_latent: int = int(self.latent_shape[0])

        # The height and width of the latent representation.
        self.h_latent: int = int(self.latent_shape[1])
        self.w_latent: int = int(self.latent_shape[2])

        # The height and width for hidden layer of the message encoder F.
        self.h_little: int = int(self.configs["h_little"])
        self.w_little: int = int(self.configs["w_little"])
        self.c_little: int = int(self.configs["c_little"])

        # The neural network that transforms messages into a latent offset.
        self.message_encoder = self.setup_message_encoder().to(self.device)

    def setup_message_encoder(self) -> nn.Module:
        """
        Sets up the neural network that turns messages into latent space representations.
        """
        # R^ell -> latent space. This is what encodes the messages.
        # Project the message into a (c_little, h_little, w_little) seed map...
        layers = [
            nn.Linear(self.message_length, self.h_little * self.w_little * self.c_little),
            nn.SiLU(),
            Reshape(self.c_little, self.h_little, self.w_little),
        ]

        # ...then double the spatial size each step until it matches the latent grid.
        num_upsamples = int(np.log2(self.h_latent / self.h_little))
        assert self.h_little * 2 ** num_upsamples == self.h_latent
        assert self.w_little * 2 ** num_upsamples == self.w_latent
        for _ in range(num_upsamples):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(self.c_little, self.c_little, 3, padding=1),
                nn.SiLU(),
            ]

        # Final conv fixes the channel count to the latent's. Zero-init means the
        # offset starts at zero, so the stego latent equals the cover latent until
        # the network learns to embed (RoSteALS).
        final_conv = nn.Conv2d(self.c_little, self.c_latent, 1)
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)
        layers.append(final_conv)

        return nn.Sequential(*layers)

    def encode_batch(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Watermarks a batch of images, fully in torch and differentiable end to end.
        This is the interface used during training.

        Args:
            cover: a (B, C, H, W) float tensor in [0, 1] on ``self.device``.
            message: a (B, message_length) float tensor on ``self.device``.

        Returns:
            The watermarked images as a (B, C, H, W) tensor (unclamped [0, 1]).
        """
        # delta is the latent-space offset that carries the message (RoSteALS).
        delta = self.message_encoder(message)

        # the cover image placed in latent space.
        cover_latent = self.image_autoencoder.encode(cover)
        assert delta.shape == cover_latent.shape

        # add the offset, then decode the watermarked latent back into image space.
        return self.image_autoencoder.decode(cover_latent + delta)

    def encode_image(self, cover: np.ndarray, message: np.ndarray) -> np.ndarray:
        """
        Single-image numpy convenience wrapper around :meth:`encode_batch`, for
        inference and visualization. Training should call ``encode_batch`` directly.
        """
        assert cover.shape == (self.h_image, self.w_image, self.c_image)
        assert message.shape == (self.message_length, 1)

        # numpy -> torch: (H, W, C) -> (1, C, H, W), (L, 1) -> (1, L), onto device.
        cover_t = torch.from_numpy(cover).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        message_t = torch.from_numpy(message).float().reshape(1, self.message_length).to(self.device)

        with torch.no_grad():
            watermarked = self.encode_batch(cover_t, message_t)

        # torch -> numpy: clamp to valid pixels, (1, C, H, W) -> (H, W, C).
        watermarked = watermarked.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0)
        return watermarked.cpu().numpy()

    def decode_image(self, image: np.ndarray) -> np.ndarray:
        # TODO
        pass

    def train(self) -> None:
        # TODO
        pass
