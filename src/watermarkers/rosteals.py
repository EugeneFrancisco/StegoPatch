"""
This file defines an image watermarker using RoSteALS method from Bui et al.
"""
import os
from collections import deque
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from src.watermarkers.image_watermarker import ImageWatermarker
from src.autoencoders.vqgan import VQGAN
from tqdm import tqdm
import src.utils as utils
from src.noisers.noiser import Noiser

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

        # ===== Noising material ======

        # The noiser which we will ultimately use to apply noise between creating stego images
        # and decoding messages.
        self.noiser: Noiser = self.configs["noiser"]

        # A flag for when to begin applying noise.
        self.begin_noising: bool = False

        # ===== Dataset material ======

        # The full training dataset, which should have the same size as training_data_size
        self.dataset: Dataset = self.configs["dataset"]

        # The size of the initial exposure set (should be one or two minibatches)
        self.first_exposure_set_size: int = self.configs["training_data_sizes"][0]

        # The size of the second exposure set (should be a significant portion of the data)
        self.second_exposure_set_size: int = self.configs["training_data_sizes"][1]

        # The size of the full training data.
        self.training_data_size: int = self.configs["training_data_sizes"][2]

        # ===== Training Hyperparameters are below =======

        # AdamW learning rate
        self.lr: float = self.configs["learning_rate"]

        # The number of training epochs.
        self.num_epochs: int = self.configs["num_epochs"]

        # The number of training epochs before we expose the model to the full training set.
        self.num_epochs_for_small_batch: int = self.configs["num_epochs_for_small_batch"]

        # The batch size.
        self.batch_size: int = self.configs["batch_size"]

        # Controls the weight of the MSE loss for the quality loss objective.
        self.alpha: float = self.configs["alpha"]

        # Controls the weight of the quality loss objective.
        self.beta: float = self.configs["beta_min"]

        # Once we begin scheduling beta, we will linearly increase from beta to
        # beta_max by beta_delta
        self.beta_max: float = self.configs["beta_max"]

        # The offset we apply to delta each time we add to it.
        self.beta_delta: float = self.configs["beta_delta"]

        # An update flag for when we can begin linearly increasing beta.
        self.update_flag: bool = False

        # Whether or not we will keep a tensorboard to track progress.
        self.log_tensorboard: bool = self.configs["log_tensorboard"]

        # The TensorBoard writer used to log losses during training. Only created
        # when logging is enabled.
        self.tensorboard = None
        if self.log_tensorboard:
            # Where TensorBoard training logs are written. The run timestamp is
            # appended so each run gets its own subdirectory rather than overwriting
            # the last.
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base_log_dir = self.configs.get("tensorboard_log_dir", "runs/rosteals")
            self.tensorboard_log_dir: str = f"{base_log_dir}_{run_timestamp}"
            self.tensorboard = SummaryWriter(log_dir=self.tensorboard_log_dir)
            print("Logging to tensoboard.")
        else:
            print("Not logging to tensorboard.")

        # AdamW optimizes both trainable networks (the frozen autoencoder is excluded).
        # Kept as a member so optimizer state persists across train_until calls.
        self.optimizer = torch.optim.AdamW(
            list(self.message_encoder.parameters())
            + list(self.secret_decoder.parameters()),
            lr=self.lr,
        )

        # Global training step, used for TensorBoard logging across train_until calls.
        self.step = 0

    def setup_message_encoder(self) -> nn.Module:
        """
        Sets up the neural network that turns messages into latent space representations.
        """
        # R^ell -> latent space. This is what encodes the messages.
        # Project the message into the latent space.
        layers = [
            nn.Linear(self.message_length, self.h_little * self.w_little * self.c_little),
            nn.SiLU(),
            nn.Unflatten(1, (self.c_little, self.h_little, self.w_little)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.c_little, self.c_little, 3, padding = 1)
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
        # Start from ImageNet-pretrained weights so the decoder inherits useful
        # low-level features instead of learning everything from scratch. The
        # pretrained weights were trained on ImageNet-normalized inputs, but we do
        # not re-apply that normalization: decode_batch feeds the decoder raw
        # ~[0, 1] pixels and lets fine-tuning adapt the weights to that range.
        decoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

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

        # Track how large the offset is getting during training: if bit accuracy is
        # stuck, this tells us whether the encoder is actually learning to embed.
        if self.log_tensorboard and self.message_encoder.training:
            delta_l2 = deltas.flatten(1).norm(dim=1).mean().item()
            self.tensorboard.add_scalar("delta_l2", delta_l2, self.step)

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
        Recovers the message logits from a batch of images.
        This is the interface used during training.

        The stego images come straight from ``encode_batch`` (the unclamped VQGAN
        decode output), so pixels are approximately in [0, 1] but may overshoot
        slightly past either end; they are passed to the decoder as-is.
        Args:
            stego_images: a (B, C, H, W) float tensor of ~[0, 1] pixels on
                ``self.device``.
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

    def train(self) -> None:
        """
        Trains the message encoder and secret decoder on a dataset of cover images.
        The frozen image autoencoder is used as-is; only the message encoder and
        secret decoder are optimized.
        """
        assert len(self.dataset) == self.training_data_size

        self.secret_decoder.train()
        self.message_encoder.train()

        # Timestamp of when training started, used to name the saved weights.
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Where checkpoints are written (a Modal Volume path when run remotely).
        models_dir = self.configs.get("models_dir", "models")

        # =================== Checkpoint 0 begins =================
        # Train one or two mini batches of data until the 0.9 bit accuracy threshold is reached.

        baby_dataset = Subset(
                            self.dataset,
                            range(min(self.first_exposure_set_size, len(self.dataset)))
                            )

        # Train until bit accuracy is 0.9. The baby_dataset contains only a couple minibatches
        # worth of images so the max_epochs here is large because we want to do many passes over
        # those images.
        self.train_until(
                        baby_dataset,
                        bit_accuracy_threshold=0.9,
                        max_epochs=self.num_epochs_for_small_batch
                        )
        self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint1.pt")

        # =================== Checkpoint 1 begins =================
        # Expose the model to a much larger portion of training data (around half of total)
        second_dataset = Subset(
            self.dataset,
            range(min(self.second_exposure_set_size, len(self.dataset)))
        )
        self.train_until(
                        second_dataset,
                        bit_accuracy_threshold=0.8,
                        max_epochs=self.num_epochs
                        )
        self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint2.pt")


        # =================== Checkpoint 2 begins =================
        # Train on same dataset as before but begin updating beta.
        # Train until bit accuracy is 0.98.
        self.update_flag = True
        self.train_until(
                        second_dataset,
                        bit_accuracy_threshold=0.95,
                        save_every_epoch=True,
                        start_time=start_time,
                        checkpoint=3
                        )
        self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint3.pt")

        # =================== Checkpoint 3 begins =================
        # Expose the model to the full dataset.
        self.train_until(
                        self.dataset,
                        bit_accuracy_threshold=0.98,
                        save_every_epoch=True,
                        start_time=start_time,
                        checkpoint=4
                        )
        self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint4.pt")

        # =================== Checkpoint 4 begins =================
        self.begin_noising = True
        self.train_until(self.dataset, max_epochs = 5)
        self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint5.pt")

    def restart_training(self, save_path: str, checkpoint: int) -> None:
        """
        Restarts training from the passed in save_path and starting from the checkpoint.
        Args:
            save_path: a path to a .pt file that saves the training information.
            checkpoint: an int that is either 1, 2, 3, or 4. If it is 1, this means we restart
            training from the point that we reveal the model to the half the training set (t1)
            If it is 2, then we start training from the point that we begin incrementing beta,
            meaning the quality loss gets increasingly weighted. If it is 3, then we begin training
            where we reveal the model to the full training set. If it is 4, then we begin adding
            noise between watermark insertion and message decoding.
        """
        assert checkpoint in (1, 2, 3, 4)

        # Restore weights, optimizer state, and the beta schedule progress.
        self.load_model(save_path)

        self.secret_decoder.train()
        self.message_encoder.train()

        # Timestamp of when this resumed run started, used to name new checkpoints.
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        models_dir = self.configs.get("models_dir", "models")

        if checkpoint == 1:
            # =================== Checkpoint 1 begins =================
            # Train until bit accuracy is 0.80 again. Update_flag is false so that
            # we prioritize recovery still. At this point, we are exposing the model
            # to the 50,000 of the training examples.
            self.update_flag = False
            assert len(self.dataset) == self.second_exposure_set_size
            self.train_until(
                            self.dataset,
                            bit_accuracy_threshold=0.80,
                            progress_bar="step",
                            )
            self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint2.pt")

        if checkpoint in [1, 2]:
            # =================== Checkpoint 2 begins =================
            # Start incrementing beta.
            self.update_flag = True
            assert len(self.dataset) == self.second_exposure_set_size
            self.train_until(
                            self.dataset,
                            bit_accuracy_threshold=0.95,
                            progress_bar="step",
                            )
            self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint3.pt")

        if checkpoint in [1, 2, 3]:
            # =================== Checkpoint 3 begins =================
            # Train on all 100,000 examples now until 0.98 is reached.
            self.beta = self.beta_max
            assert len(self.dataset) == self.training_data_size
            self.train_until(
                            self.dataset,
                            bit_accuracy_threshold=0.98,
                            progress_bar="step"
                            )
            self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint4.pt")

        # ====================== Checkpoint 4 begins ==============
        # Begin training with noise
        assert len(self.dataset) == self.training_data_size
        self.beta = self.beta_max

        self.begin_noising = True
        self.train_until(
                        self.dataset,
                        max_epochs=5,
                        progress_bar="step",
                        save_every_epoch=True,
                        start_time=start_time,
                        checkpoint=5
                        )
        self.save_model(f"{models_dir}/rosteals_{start_time}/checkpoint5.pt")


    def save_model(self, path: str = "models/rosteals.pt") -> None:
        """
        Saves the trainable network weights (message encoder and secret decoder)
        to ``path``. The frozen autoencoder is not saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "message_encoder": self.message_encoder.state_dict(),
                "secret_decoder": self.secret_decoder.state_dict(),
                # Optimizer state (AdamW moments + step counts) so training can
                # resume without restarting the optimizer cold.
                "optimizer": self.optimizer.state_dict(),
                # Training progress that mutates across train_until calls and is
                # needed to pick the beta schedule back up exactly where we left off.
                "beta": self.beta,
                "update_flag": self.update_flag,
                "step": self.step,
            },
            path,
        )

    def load_model(self, path: str = "models/rosteals.pt") -> None:
        """
        Loads previously saved message encoder and secret decoder weights from
        ``path`` into this instance.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.message_encoder.load_state_dict(checkpoint["message_encoder"])
        self.secret_decoder.load_state_dict(checkpoint["secret_decoder"])

        # Restore training state if present, so we can resume rather than just
        # run inference. Guarded with .get so older weight-only checkpoints
        # still load.
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.beta = checkpoint.get("beta", self.beta)
        self.update_flag = checkpoint.get("update_flag", self.update_flag)
        self.step = checkpoint.get("step", self.step)

    def train_until(
            self,
            dataset: Dataset,
            bit_accuracy_threshold=None,
            max_epochs=None,
            save_every_epoch=False,
            progress_bar="epoch",
            start_time=None,
            checkpoint=None,
        ) -> None:
        """
        Trains the message encoder and secret decoder on the passed in dataset
        until theshold is reached or max_epochs is reached.

        If ``save_every_epoch`` is True, the model is saved after each epoch finishes.
        When ``start_time`` is provided, those per-epoch checkpoints are written next
        to the ``train``/``restart_training`` checkpoints (``{models_dir}/rosteals_
        {start_time}/checkpoint{checkpoint}_epoch_{n}.pt``) so they share the same run
        directory and are grouped by the checkpoint phase that produced them;
        otherwise the default ``save_model`` path is used. ``checkpoint`` is the number
        of the named checkpoint this training phase culminates in.

        ``progress_bar`` controls where the tqdm bar lives: "epoch" (default) wraps
        the outer epoch loop, "step" wraps the inner per-batch loop within each epoch.
        """
        assert progress_bar in ("epoch", "step")
        max_epochs = max_epochs if max_epochs is not None else self.num_epochs
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Rolling buffer of the previous 10 bit accuracies, used to smooth out
        # noise before deciding whether the threshold has been reached.
        recent_bit_accuracies = deque(maxlen=10)

        epochs = range(max_epochs)
        if progress_bar == "epoch":
            epochs = tqdm(epochs, desc="epochs")

        for epoch in epochs:
            steps = tqdm(loader, desc="steps") if progress_bar == "step" else loader
            for covers in steps:
                covers = covers.to(self.device)

                # Random {0, 1} messages, one per cover in the batch.
                messages = torch.randint(
                    0, 2, (covers.shape[0], self.message_length), device=self.device
                ).float()

                stego_images = self.encode_batch(covers, messages)

                # Quality is always measured against the clean watermarked image:
                # the noise layer is a channel corruption used to train decoder
                # robustness, not something the encoder should compensate for.
                loss_quality = self.get_quality_loss(covers, stego_images)

                # Apply noise (if we are at that stage) only to what the decoder
                # sees, so robustness training does not leak into the quality loss.
                decoder_input = stego_images
                if self.begin_noising:
                    decoder_input = self.noiser.apply_noise(stego_images)

                recovered_messages = self.decode_batch(decoder_input)

                loss_recovery = self.get_recovery_loss(messages, recovered_messages)
                loss = loss_recovery + self.beta * loss_quality

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Hard-decode the logits and measure the fraction of correct bits.
                predicted_bits = (recovered_messages > 0).float()
                bit_accuracy = (predicted_bits == messages).float().mean().item()
                recent_bit_accuracies.append(bit_accuracy)

                if self.log_tensorboard:
                    self.tensorboard.add_scalar("loss/recovery", loss_recovery.item(), self.step)
                    self.tensorboard.add_scalar("loss/quality", loss_quality.item(), self.step)
                    self.tensorboard.add_scalar("bit_accuracy", bit_accuracy, self.step)
                    self.tensorboard.add_scalar("beta", self.beta, self.step)
                self.step += 1

                # Only stop once the rolling average over the last 10 batches
                # (once the buffer is full) beats the threshold.
                if (
                    bit_accuracy_threshold is not None
                    and len(recent_bit_accuracies) == recent_bit_accuracies.maxlen
                ):
                    rolling_average = sum(recent_bit_accuracies) / len(recent_bit_accuracies)
                    if rolling_average >= bit_accuracy_threshold:
                        return
                self.update_beta()

            if save_every_epoch:
                if start_time is not None:
                    # Land per-epoch checkpoints in the same run directory the
                    # train/restart_training named checkpoints use, prefixed with
                    # the checkpoint phase and tagged with the epoch they
                    # correspond to (1-indexed) so they don't collide across phases.
                    models_dir = self.configs.get("models_dir", "models")
                    self.save_model(
                        f"{models_dir}/rosteals_{start_time}/"
                        f"checkpoint{checkpoint}_epoch_{epoch + 1}.pt"
                    )
                else:
                    self.save_model()

    def update_beta(self):
        """
        Linearly increases beta once the flag is set and until beta reaches
        the max beta.
        """
        if not self.update_flag or self.beta >= self.beta_max:
            return
        self.beta += self.beta_delta

    def get_quality_loss(
            self,
            covers: torch.Tensor,
            stego_images: torch.Tensor,
        ) -> torch.Tensor:
        """
        Return the quality loss on the passed in data.
        Args:
            covers: a (B, C, H, W) tensor of images that were used to create watermarks.
            stego_images: a (B, C, H, W) tensor of images that have been watermarked with the passed
                in messages.
        Returns:
            A tuple of (recovery loss, quality loss).
        """

        # Calculate the MSE loss between the covers and thT stego_images.
        covers_yuv = utils.rgb_to_yuv(covers)
        stego_images_yuv = utils.rgb_to_yuv(stego_images)
        loss_mse = nn.functional.mse_loss(stego_images_yuv, covers_yuv)

        # Calculate the LPIPS loss between the covers and the stego images.
        loss_lpips = utils.lpips_loss(covers, stego_images)

        loss_quality = loss_lpips + self.alpha * loss_mse
        return loss_quality

    def get_recovery_loss(
            self,
            messages: torch.Tensor,
            recovered_messages: torch.Tensor
        ) -> torch.Tensor:
        """
        Return the recovery loss on the passed in data.
        Args:
            messages: a (B, message_length) tensor of messages that were encoded.
            recovered_messages: a (B, message_length) tensor of recovered messages from the
                watermarked images.
        Returns:
            The torch tensor with the recovery loss.
        """
        loss_recovery = utils.bce_loss(recovered_messages, messages)
        return loss_recovery
