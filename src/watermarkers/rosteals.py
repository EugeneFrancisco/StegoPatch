"""
This file defines an image watermarker using RoSteALS method from Bui et al.
"""
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from src.watermarkers.image_watermarker import ImageWatermarker
from src.autoencoders.vqgan import VQGAN
import src.utils as utils

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

        # The ResNet-50 that recovers the message from a (watermarked) image.
        self.secret_decoder = self.setup_secret_decoder().to(self.device)

        # ===== Training Hyperparameters are below =======

        # The number of training epochs.
        self.num_epochs: int = self.configs["num_epochs"]

        # The batch size.
        self.batch_size: int = self.configs["batch_size"]

        # Controls the weight of the MSE loss for the quality loss objective.
        self.alpha: float = self.configs["alpha"]

        # Controls the weight of the quality loss objective.
        self.beta: float = self.configs["beta"]

        # Where TensorBoard training logs are written.
        self.tensorboard_log_dir: str = self.configs.get("tensorboard_log_dir", "runs/rosteals")

        # The TensorBoard writer used to log losses during training.
        self.tensorboard = SummaryWriter(log_dir=self.tensorboard_log_dir)

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

    def setup_secret_decoder(self) -> nn.Module:
        """
        Sets up the ResNet-50 secret decoder D (Bui et al.): it maps an image back
        to a length-``message_length`` vector of logits, one per message bit.
        """
        # Train the decoder from scratch (no ImageNet weights) so it learns to read
        # the watermark rather than ImageNet features.
        decoder = resnet50(weights=None)

        # ResNet-50's stem expects 3 channels; widen it if our images differ.
        if self.c_image != 3:
            decoder.conv1 = nn.Conv2d(
                self.c_image, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace the 1000-way ImageNet head with one logit per message bit.
        decoder.fc = nn.Linear(decoder.fc.in_features, self.message_length)
        return decoder

    def encode_batch(self, covers: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Watermarks a batch of images, fully in torch and differentiable end to end.
        This is the interface used during training.

        Args:
            covers: a (B, C, H, W) float tensor in [0, 1] on ``self.device``.
            messages: a (B, message_length) float tensor on ``self.device``.

        Returns:
            The watermarked images as a (B, C, H, W) tensor (unclamped [0, 1]).
        """
        # delta is the latent-space offset that carries the message (RoSteALS).
        deltas = self.message_encoder(messages)

        # the cover image placed in latent space.
        covers_latent = self.image_autoencoder.encode(covers)
        assert deltas.shape == covers_latent.shape

        # add the offset, then decode the watermarked latent back into image space.
        return self.image_autoencoder.decode(covers_latent + deltas)

    def encode_image(self, cover: np.ndarray, message: np.ndarray) -> np.ndarray:
        """
        Single-image numpy convenience wrapper around :meth:`encode_batch`, for
        inference and visualization. Training should call ``encode_batch`` directly.
        """
        assert cover.shape == (self.h_image, self.w_image, self.c_image)
        assert message.shape == (self.message_length, 1)

        # numpy -> torch: (H, W, C) -> (1, C, H, W), (L, 1) -> (1, L), onto device.
        cover_t = torch.from_numpy(cover).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        message_t = torch.from_numpy(
            message
        ).float().reshape(1, self.message_length).to(self.device)

        with torch.no_grad():
            watermarked = self.encode_batch(cover_t, message_t)

        # torch -> numpy: clamp to valid pixels, (1, C, H, W) -> (H, W, C).
        watermarked = watermarked.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0)
        return watermarked.cpu().numpy()

    def decode_batch(self, stego_images: torch.Tensor) -> torch.Tensor:
        """
        Recovers the message logits from a batch of images, fully in torch and
        differentiable end to end. This is the interface used during training.

        Args:
            stego_images: a (B, C, H, W) float tensor in [0, 1] on ``self.device``.

        Returns:
            A (B, message_length) tensor of raw logits (apply a sigmoid for
            per-bit probabilities, or threshold at 0 for hard bits).
        """
        return self.secret_decoder(stego_images)

    def decode_image(self, stego_image: np.ndarray) -> np.ndarray:
        """
        Single-image numpy convenience wrapper around :meth:`decode_batch`, for
        inference. Returns the predicted message as a (message_length, 1) array of
        0/1 bits, matching the message shape expected by :meth:`encode_image`.
        """
        assert stego_image.shape == (self.h_image, self.w_image, self.c_image)

        # numpy -> torch: (H, W, C) -> (1, C, H, W), onto device.
        image_t = torch.from_numpy(
            stego_image
        ).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.decode_batch(image_t)

        # Positive logit -> bit 1. (1, L) -> (L, 1).
        bits = (logits.squeeze(0) > 0).float().reshape(self.message_length, 1)
        return bits.cpu().numpy()

    def train(self, images: torch.Tensor) -> None:
        """
        Trains the message encoder and secret decoder on a dataset of cover images.
        The frozen image autoencoder is used as-is; only the message encoder and
        secret decoder are optimized.

        Args:
            images: a (N, C, H, W) float tensor in [0, 1] of cover images.
        """
        self.message_encoder.train()
        self.secret_decoder.train()

        # Clear any existing TensorBoard data so this run starts fresh.
        self.tensorboard.close()
        if os.path.exists(self.tensorboard_log_dir):
            shutil.rmtree(self.tensorboard_log_dir)
        self.tensorboard = SummaryWriter(log_dir=self.tensorboard_log_dir)

        optimizer = torch.optim.Adam(
            list(self.message_encoder.parameters())
            + list(self.secret_decoder.parameters())
        )

        num_images = images.shape[0]
        step = 0
        for epoch in range(self.num_epochs):
            # Shuffle the images at the start of each epoch.
            perm = torch.randperm(num_images)
            for start in range(0, num_images, self.batch_size):
                covers = images[perm[start:start + self.batch_size]].to(self.device)

                # Sample random binary messages, one per cover in the batch.
                messages = torch.randint(
                    0, 2, (covers.shape[0], self.message_length), device=self.device
                ).float()

                # Watermark, then try to recover the message.
                stego_images = self.encode_batch(covers, messages)
                recovered_messages = self.decode_batch(stego_images)

                recovery_loss, quality_loss = self.get_loss(
                    covers, messages, stego_images, recovered_messages
                )

                total_loss = recovery_loss + self.beta * quality_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Log the losses for this batch to TensorBoard.
                self.tensorboard.add_scalar("loss/recovery", recovery_loss.item(), step)
                self.tensorboard.add_scalar("loss/quality", quality_loss.item(), step)
                self.tensorboard.add_scalar("loss/total", total_loss.item(), step)
                step += 1

            print(f"Epoch {epoch + 1}/{self.num_epochs}, loss: {total_loss.item():.4f}")

    def get_loss(
            self,
            covers: torch.Tensor,
            messages: torch.Tensor,
            stego_images: torch.Tensor,
            recovered_messages: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple of (recovery loss, quality loss).
        Args:
            covers: a (B, C, H, W) tensor of images that were used to create watermarks.
            messages: a (B, message_length) tensor of messages that were encoded.
            stego_images: a (B, C, H, W) tensor of images that have been watermarked with the passed
                in messages.
            recovered_messages: a (B, message_length) tensor of recovered messages from the
                watermarked images.

        Returns:
            A tuple of (recovery loss, quality loss).
        """

        # Calculate the MSE loss between the covers and the stego_images.
        covers_yuv = utils.rgb_to_yuv(covers)
        stego_images_yuv = utils.rgb_to_yuv(stego_images)
        loss_mse = nn.functional.mse_loss(stego_images_yuv, covers_yuv)

        # Calculate the LPIPS loss between the covers and the stego images.
        loss_lpips = utils.lpips_loss(covers, stego_images)

        loss_quality = loss_lpips + self.alpha * loss_mse

        # Calculate the recovery loss.
        loss_recovery = utils.bce_loss(recovered_messages, messages)
        return (loss_recovery, loss_quality)
