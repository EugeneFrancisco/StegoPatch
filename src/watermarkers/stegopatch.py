"""
This file defines the StegoPatch Watermarking class.
"""
import os
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

import src.utils as utils
from src.watermarkers.image_watermarker import ImageWatermarker
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.vqgan import VQGAN
from src.noisers.noiser import Noiser

class StegoPatch(ImageWatermarker):
    """
    This class defines the Stego Patch watermarker which is a variant of RoSteALS in that it
    watermarks images in patches. Specifically, it watermarks an cover c and message m in the
    following way. First, C is split up into "patches" of a fixed square side. Each patch p
    is watermarked using the same technique as in RoSteALS: it is passed through an autoencoder E
    to retrive a latent variable z_p = E(p). A message encoder F outputs delta = F(m) and we form
    a "watermarked latent" z_p + delta. This is passed through the autoencoder generator G to form
    the watermarked patch p_hat = G(z_p + delta). The watermarked patches are stitched together
    to form the watermarked image c_hat. To decode the message, a secret decoder D takes in c_hat
    and outputs a predicted message m_hat.
    """
    def __init__(self, configs):
        super().__init__(configs)

        self.device: str = configs["device"]

        # The patch size that will be used for watermarking
        self.patch_size: int = self.configs["patch_size"]

        self.image_autoencoder: AutoEncoder = VQGAN(device = self.device)
        self.latent_shape: tuple[int] = VQGAN.get_latent_dim(self.patch_size, self.patch_size)

        # The number of channels in the latent representation.
        self.c_latent: int = int(self.latent_shape[0])

        # The height and width of the latent representation.
        self.h_latent: int = int(self.latent_shape[1])
        self.w_latent: int = int(self.latent_shape[2])

        # The height and width for hidden layer of the message encoder F.
        self.h_little: int = int(self.configs["h_little"])
        self.w_little: int = int(self.configs["w_little"])
        self.c_little: int = int(self.configs["c_little"])

        self.message_encoder: nn.Module = self.setup_message_encoder().to(self.device)

        self.secret_decoder: nn.Module = self.setup_secret_decoder().to(self.device)

        # ============ Noising Material ===========

        # The noiser which we will ultimately use to apply noise between creating stego images
        # and decoding messages.
        self.noiser: Noiser = self.configs["noiser"]

        # A flag for when to begin applying the full noise blend during training.
        self.begin_noising: bool = False

        # A flag for when to begin applying cropping noise. Unlike begin_noising,
        # cropping is turned on from the very first checkpoint: robustness to
        # cropping is the whole point of StegoPatch, so the model trains against
        # crops before any other corruption is introduced.
        self.begin_cropping: bool = False

        # Probability that a batch is cropped while begin_cropping is on (and the
        # full noise blend is still off).
        self.crop_probability: float = 0.5

        # ========== Dataset Material =============

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

        # The batch size.
        self.batch_size: int = self.configs["batch_size"]

        # The number of epochs to run before stopping a checkpoint. Typically we won't run this
        # many epochs because the bit accuracy threshold should be crossed before this happens.
        self.num_epochs = self.configs["num_epochs"]

        # The number of epochs to run on the first (tiny) exposure set. That set is
        # only a minibatch or two, so a single "epoch" is just a step or two; this is
        # large to allow many passes over those few images before the threshold hits.
        self.num_epochs_for_small_batch: int = self.configs["num_epochs_for_small_batch"]

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

        # The directory where model checkpoints are written (a Modal Volume path
        # when run remotely). Required: fail loudly if it isn't configured.
        self.models_dir: str = self.configs["models_dir"]

        # ====== validation material =========
        self.test_set = self.configs.get("test_set", None)

    def setup_message_encoder(self) -> nn.Module:
        """
        This sets up the message encoder. The message encoder is a neural network which takes in as
        input a message of length message_length and outputs a delta in the latent space that can be
        added to a patch to form a watermarked patch.
        """

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
        This sets up the secret decoder. The secret decoder is a neural network which takes as
        input a full size image and outputs a prediction for the encoded message. It does this
        in the following way. The secret decoder is a ResNet 50 loaded with imagenet weights.
        However, the tail of the ResNet (what would normally be a head that connects to
        ImageNet logits) is replaced by a 1 x 1 CNN with message_length channels. The output of the
        ResNet CNN is then an H' x W' x message_length tensor. Each element of this tensor can be
        thought of as the logits for a prediction of the encoded message. For each bit in the
        message, average the logits across all the pixels to get an average logit, making a
        message_length vector of logit averages. Finally, use these logit averages to make a message
        prediction.
        """
        # Start from ImageNet-pretrained weights so the decoder inherits useful
        # low-level features instead of learning everything from scratch. As in
        # RoSteALS, we keep the decoder operating on raw ~[0, 1] pixels and let
        # fine-tuning adapt the weights away from ImageNet normalization.
        decoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # ResNet-50's convolutional feature extractor: everything up to (but not
        # including) the global-average-pool and ImageNet head. This maps an
        # image to a (B, in_features, H', W') spatial feature map.
        in_features = decoder.fc.in_features
        backbone = nn.Sequential(
            decoder.conv1,
            decoder.bn1,
            decoder.relu,
            decoder.maxpool,
            decoder.layer1,
            decoder.layer2,
            decoder.layer3,
            decoder.layer4,
        )

        # Replace the ImageNet head with a 1 x 1 conv giving one channel per
        # message bit, so the feature map becomes (B, message_length, H', W'):
        # a per-pixel logit for every bit. AdaptiveAvgPool2d(1) then averages
        # those logits across all pixels, and Flatten drops the spatial dims to
        # leave a (B, message_length) vector of averaged logits.
        return nn.Sequential(
            backbone,
            nn.Conv2d(in_features, self.message_length, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def encode_image(self, cover: np.ndarray, message: np.ndarray) -> np.ndarray:
        cover_t = torch.from_numpy(cover)
        cover_t = cover_t.unsqueeze(0)
        cover_t = cover_t.permute(0, 3, 1, 2)
        cover_t = cover_t.to(self.device)
        message = message.reshape(1, self.message_length)
        message_t = torch.from_numpy(message).float()
        message_t = message_t.to(self.device)
        stego = self.encode_batch(cover_t, message_t)
        stego = stego.squeeze(0)
        stego = stego.permute(1, 2, 0)
        return stego.detach().cpu().numpy()

    def encode_batch(self, covers: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Args:
            covers: a (B, C, H, W) tensor of images to watermark.
            messages: a (B, message_length) tensor of messages to use.
        Returns:
            A (B, C, H, W) tensor of images that have been watermarked.
        """

        B, C, H, W = covers.shape

        assert H == W
        assert H % self.patch_size == 0

        # Number of patches along each spatial axis (images are square so these are equal).
        nh = H // self.patch_size
        nw = W // self.patch_size

        # First, "unravel" each cover into its patches and stack every image's patches
        # together, giving a tensor of shape (B * nh * nw, C, patch_size, patch_size).
        #
        # Split each spatial axis into (tile_index, within_tile_offset), move the tile
        # indices next to the batch dim, then flatten (B, nh, nw) into one leading axis.
        # The (nh, nw) ordering is row-major (patch (i, j) lands at flat index
        # i * nw + j within each image), which we rely on to stitch patches back together.
        patches = covers.reshape(B, C, nh, self.patch_size, nw, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B * nh * nw, C, self.patch_size, self.patch_size)

        # Now placed in the latent space.
        patches_latent = self.image_autoencoder.encode(patches)

        # Number of patches per image.
        num_patches = nh * nw

        deltas = self.message_encoder(messages)
        # Track how large the offset is getting during training: if bit accuracy is
        # stuck, this tells us whether the encoder is actually learning to embed.
        if self.log_tensorboard and self.message_encoder.training:
            delta_l2 = deltas.flatten(1).norm(dim=1).mean().item()
            self.tensorboard.add_scalar("delta_l2", delta_l2, self.step)

        # Repeat each image's delta across all of its patches so the deltas line up with
        # the patch ordering above. Because image b's patches occupy the contiguous block
        # [b*num_patches, (b+1)*num_patches), we repeat *consecutively* (repeat_interleave),
        # giving shape (B * num_patches, *latent_dims) where row b*num_patches + k is image
        # b's delta for every patch k.
        deltas = deltas.repeat_interleave(num_patches, dim=0)

        assert deltas.shape == patches_latent.shape

        watermarked_patches_latent = patches_latent + deltas
        watermarked_patches = self.image_autoencoder.decode(watermarked_patches_latent)

        # Stitch the watermarked patches back into full images. This inverts the unravel
        # above: split the flat patch axis back into (B, nh, nw), move the spatial tile
        # indices back next to their within-patch offsets, then merge each (nh, patch_size)
        # and (nw, patch_size) pair into the full H and W.
        watermarked = watermarked_patches.reshape(B, nh, nw, C, self.patch_size, self.patch_size)
        watermarked = watermarked.permute(0, 3, 1, 4, 2, 5)
        watermarked = watermarked.reshape(B, C, H, W)

        return watermarked

    def decode_image(self, stego_image: np.ndarray) -> np.ndarray:
        stego_image_t = torch.from_numpy(stego_image)
        stego_image_t = stego_image_t.permute(0, 3, 1, 2)
        stego_image_t = stego_image_t.to(self.device)
        message = self.decode_batch(stego_image_t)
        return message.detach().cpu().numpy()

    def decode_batch(self, stego_images: torch.Tensor) -> torch.Tensor:
        """
        Returns the decoded messages from a batch of stego images.
        """
        return self.secret_decoder(stego_images)

    def train(self) -> None:
        """
        Trains the message encoder and secret decoder via curriculum learning,
        closely mirroring :meth:`RoSteALS.train`. The one structural difference is
        that cropping robustness is trained from the very first checkpoint (the
        whole purpose of StegoPatch), while every other noise type is withheld
        until the final checkpoint. The frozen image autoencoder is used as-is;
        only the message encoder and secret decoder are optimized.
        """
        assert len(self.dataset) == self.training_data_size

        self.secret_decoder.train()
        self.message_encoder.train()

        # Timestamp of when training started, used to name the saved weights.
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Crop (with probability self.crop_probability) from checkpoint 0 onward;
        # all other noise types stay off until checkpoint 4.
        self.begin_cropping = True

        # =================== Checkpoint 0 begins =================
        # Train one or two mini batches of data until the 0.9 bit accuracy threshold is reached.
        # The baby_dataset contains only a couple minibatches worth of images so the max_epochs
        # here is large because we want to do many passes over those images.
        baby_dataset = Subset(
            self.dataset,
            range(min(self.first_exposure_set_size, len(self.dataset)))
        )
        self.train_until(
            baby_dataset,
            bit_accuracy_threshold=0.9,
            max_epochs=self.num_epochs_for_small_batch,
            save_every_epoch=True,
            start_time=start_time,
            checkpoint=0
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint1.pt")

        # =================== Checkpoint 1 begins =================
        # Expose the model to a much larger portion of the training data (around half of total)
        # and train until bit accuracy crosses 0.8.
        second_dataset = Subset(
            self.dataset,
            range(min(self.second_exposure_set_size, len(self.dataset)))
        )
        self.train_until(
            second_dataset,
            bit_accuracy_threshold=0.8,
            max_epochs=self.num_epochs,
            save_every_epoch=True,
            start_time=start_time,
            checkpoint=1
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint2.pt")

        # =================== Checkpoint 2 begins =================
        # Train on the same dataset but begin ramping beta from beta_min to beta_max, weighting
        # quality more heavily. Train until bit accuracy reaches 0.95.
        self.update_flag = True
        self.train_until(
            second_dataset,
            bit_accuracy_threshold=0.95,
            save_every_epoch=True,
            start_time=start_time,
            checkpoint=2
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint3.pt")

        # =================== Checkpoint 3 begins =================
        # Expose the model to the full training set and train until bit accuracy reaches 0.98.
        self.train_until(
            self.dataset,
            bit_accuracy_threshold=0.98,
            save_every_epoch=True,
            start_time=start_time,
            checkpoint=3,
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint4.pt")

        # =================== Checkpoint 4 begins =================
        # Turn on the full noise blend (differentiable + imagenet-c corruptions, alongside
        # cropping) to finish training for robustness.
        self.begin_noising = True
        self.train_until(
            self.dataset,
            max_epochs=5,
            save_every_epoch=True,
            start_time=start_time, 
            checkpoint=4
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint5.pt")

    def restart_training(self, save_path: str, checkpoint: int) -> None:
        """
        Resumes training from the weights saved at ``save_path``, picking the
        curriculum back up at the given ``checkpoint`` phase and running every
        phase from there to the end. Mirrors :meth:`train` but skips the phases
        that precede ``checkpoint``.

        Args:
            save_path: path to a .pt file saved by :meth:`save_model` (weights,
                optimizer state, and beta-schedule progress).
            checkpoint: which curriculum phase to resume at, one of 1, 2, 3, or 4:
                1 - reveal the model to (about half) the training set; prioritize
                    recovery (no beta ramp yet) until bit accuracy reaches 0.8.
                2 - begin ramping beta from beta_min to beta_max until 0.95.
                3 - expose the full training set until bit accuracy reaches 0.98.
                4 - turn on the full noise blend and finish training.
        """
        assert checkpoint in (1, 2, 3, 4)

        # Restore weights, optimizer state, and the beta schedule progress.
        self.load_model(save_path)

        self.secret_decoder.train()
        self.message_encoder.train()

        # Timestamp of when this resumed run started, used to name new checkpoints.
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Cropping is enabled from the very first checkpoint, so every phase we can
        # resume into (1-4) trains against crops.
        self.begin_cropping = True

        # Around half of the training data, matching the corresponding phases in train.
        second_dataset = Subset(
            self.dataset,
            range(min(self.second_exposure_set_size, len(self.dataset)))
        )

        if checkpoint == 1:
            # =================== Checkpoint 1 begins =================
            # Reveal the model to a much larger portion of the data. update_flag is
            # False so we still prioritize recovery; train until bit accuracy 0.8.
            self.update_flag = False
            self.train_until(
                second_dataset,
                bit_accuracy_threshold=0.8,
                progress_bar="step",
                save_every_epoch=True,
                start_time=start_time,
                checkpoint=1,
            )
            self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint2.pt")

        if checkpoint in (1, 2):
            # =================== Checkpoint 2 begins =================
            # Start ramping beta from beta_min to beta_max until bit accuracy 0.95.
            self.update_flag = True
            self.train_until(
                second_dataset,
                bit_accuracy_threshold=0.95,
                progress_bar="step",
                save_every_epoch=True,
                start_time=start_time,
                checkpoint=2,
            )
            self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint3.pt")

        if checkpoint in (1, 2, 3):
            # =================== Checkpoint 3 begins =================
            # Expose the model to the full training set until bit accuracy 0.98. The
            # beta ramp is finished by now, so pin beta at its max.
            self.beta = self.beta_max
            self.train_until(
                self.dataset,
                bit_accuracy_threshold=0.98,
                progress_bar="step",
                save_every_epoch=True,
                start_time=start_time,
                checkpoint=3,
            )
            self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint4.pt")

        # =================== Checkpoint 4 begins =================
        # Turn on the full noise blend (differentiable + imagenet-c, alongside
        # cropping) to finish training for robustness.
        self.beta = self.beta_max
        self.begin_noising = True
        self.train_until(
            self.dataset,
            max_epochs=5,
            progress_bar="step",
            save_every_epoch=True,
            start_time=start_time,
            checkpoint=4,
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint5.pt")

    def save_model(self, path: str) -> None:
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

    def load_model(self, path: str) -> None:
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
        until the threshold is reached or max_epochs is reached.

        If ``save_every_epoch`` is True, the model is saved after each epoch finishes.
        When ``start_time`` is provided, those per-epoch checkpoints are written next
        to the ``train`` checkpoints (``{models_dir}/stegopatch_{start_time}/
        checkpoint{checkpoint}_epoch_{n}.pt``) so they share the same run directory and
        are grouped by the checkpoint phase that produced them; otherwise the default
        ``save_model`` path is used. ``checkpoint`` is the number of the named checkpoint
        this training phase culminates in.

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

                # Quality is always measured against the clean watermarked image (as opposed to
                # the noised image).
                loss_quality = self.get_quality_loss(covers, stego_images)

                # Apply noise only to what the decoder sees.
                decoder_input = stego_images
                if self.begin_noising:
                    decoder_input = self.noiser.apply_noise(stego_images)
                elif self.begin_cropping and np.random.rand() < self.crop_probability:
                    decoder_input = self.noiser.get_noise_function("crop")(stego_images)

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
                # Per-epoch checkpoints must be explicitly located: require a
                # start_time (and checkpoint) so we always know where they land.
                assert start_time is not None and checkpoint is not None

                # Land per-epoch checkpoints in the same run directory the
                # train named checkpoints use, prefixed with the checkpoint
                # phase and tagged with the epoch they correspond to
                # (1-indexed) so they don't collide across phases.
                self.save_model(
                    f"{self.models_dir}/stegopatch_{start_time}/"
                    f"checkpoint{checkpoint}_epoch_{epoch + 1}.pt"
                )

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
            The torch tensor with the quality loss.
        """
        # Calculate the MSE loss between the covers and the stego_images.
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

    def validate(self):
        pass
