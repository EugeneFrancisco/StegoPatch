"""
This file defines the StegoPatch Watermarking class.
"""
import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

import src.utils as utils
import src.plotting.image_plotting as image_plotting
from src.watermarkers.image_watermarker import ImageWatermarker
from src.autoencoders.autoencoder import AutoEncoder
from src.autoencoders.vqgan import VQGAN
from src.noisers.stegopatch_noiser import (
    StegoPatchNoiser,
    NOISE_ROTATE,
    NOISE_JPEG_COMPRESSION,
)

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
        # and decoding messages. train_until always applies it; the curriculum controls
        # *which* corruptions are sampled by adjusting the noiser's probabilities.
        self.noiser: StegoPatchNoiser = self.configs["noiser"]

        # Probability of cropping during the early checkpoints, where cropping is the
        # only active corruption. Robustness to cropping is the whole point of
        # StegoPatch, so the model trains against crops before any other noise.
        self.crop_probability: float = 0.5

        # Per-corruption imagenet-c sampling probabilities used in the final noise
        # blend, as a {corruption_name: probability} dict. Splitting the imagenet
        # budget per corruption lets us weight e.g. jpeg over gaussian blur. Defaults
        # to an equal split of 0.225 across every corruption (the historical blend).
        self.imagenet_probabilities: dict[str, float] = self.configs.get(
            "imagenet_probabilities",
            self.noiser.uniform_imagenet_probabilities(0.225),
        )

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
        # when run remotely).
        self.models_dir: str = self.configs["models_dir"]

        # How often (in training steps) to drop a rolling auto-resume checkpoint
        # so a run that gets killed mid-training (e.g. Modal restarting it) can
        # pick back up. This is completely separate from the named / per-epoch
        # checkpoints written by save_model, which are left untouched.
        self.num_steps_to_save: int = self.configs.get("num_steps_to_save", 5_000)

        # A distinct directory (separate from the run-named checkpoints) holding
        # a single rolling checkpoint. It is overwritten every num_steps_to_save
        # steps, so only one auto-resume checkpoint ever exists at a time, and it
        # records the curriculum phase it came from so train / restart_training
        # can resume at the right place.
        self.autosave_dir: str = self.configs.get(
            "autosave_dir", f"{self.models_dir}/autosave"
        )
        self.autosave_path: str = f"{self.autosave_dir}/autosave_checkpoint.pt"

        # ====== validation material =========
        self.test_set = self.configs.get("test_set", None)
        # Validating on the full ~40k test set is slow; cap validation to the first
        # this-many images. None means use the entire test set.
        self.validation_set_size = self.configs.get("validation_set_size", None)

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
        with torch.inference_mode():
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

        B, _, H, W = covers.shape

        assert H == W
        assert H % self.patch_size == 0

        # Number of patches along each spatial axis (images are square so these are equal).
        nh = H // self.patch_size
        nw = W // self.patch_size

        # Encode the entire image to a single latent of shape (B, c_latent, H/4, W/4).
        latent = self.image_autoencoder.encode(covers)

        # The full latent's spatial dimensions, and the per-patch latent dimensions.
        # Each patch_size x patch_size pixel patch maps to an
        # h_latent x w_latent = patch_size/4 x patch_size/4 latent patch, so the same
        # nh x nw patch grid carves the latent exactly.
        _, _, Hl, Wl = latent.shape
        assert Hl == nh * self.h_latent and Wl == nw * self.w_latent

        # "Unravel" the latent into its patches and stack every image's patches together,
        # giving (B * nh * nw, c_latent, h_latent, w_latent). Split each spatial axis into
        # (tile_index, within_tile_offset), move the tile indices next to the batch dim,
        # then flatten (B, nh, nw) into one leading axis. The (nh, nw) ordering is
        # row-major, which we rely on to stitch the patches back together.
        patches_latent = latent.reshape(
            B, self.c_latent, nh, self.h_latent, nw, self.w_latent
        )
        patches_latent = patches_latent.permute(0, 2, 4, 1, 3, 5)
        patches_latent = patches_latent.reshape(
            B * nh * nw, self.c_latent, self.h_latent, self.w_latent
        )

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

        # Stitch the watermarked latent patches back into a full latent. This inverts the
        # unravel above: split the flat patch axis back into (B, nh, nw), move the spatial
        # tile indices back next to their within-patch offsets, then merge each
        # (nh, h_latent) and (nw, w_latent) pair into the full Hl and Wl.
        watermarked_latent = watermarked_patches_latent.reshape(
            B, nh, nw, self.c_latent, self.h_latent, self.w_latent
        )
        watermarked_latent = watermarked_latent.permute(0, 3, 1, 4, 2, 5)
        watermarked_latent = watermarked_latent.reshape(B, self.c_latent, Hl, Wl)

        # Decode the whole watermarked latent once, so there is no pixel-space stitching
        # boundary and the watermarked image is globally consistent.
        watermarked = self.image_autoencoder.decode(watermarked_latent)

        return watermarked

    def decode_image(self, stego_image: np.ndarray) -> np.ndarray:
        stego_image_t = torch.from_numpy(stego_image)
        stego_image_t = stego_image_t.permute(0, 3, 1, 2)
        stego_image_t = stego_image_t.to(self.device)
        logits = self.decode_batch(stego_image_t)
        logits = logits.reshape((self.message_length))
        predicted_bits = (logits > 0).long()
        return predicted_bits.detach().cpu().numpy()

    def decode_batch(self, stego_images: torch.Tensor) -> torch.Tensor:
        """
        Returns the decoded messages from a batch of stego images.
        """
        return self.secret_decoder(stego_images)

    def decode_file(self, path: str, size: int | None = None) -> np.ndarray:
        """
        Decodes the message embedded in a watermarked PNG and returns it as a numpy
        array of bits.

        Args:
            path: file path to a .png image.
            size: if given, the image is resized to a ``size`` x ``size`` square
                before decoding. Decoding is most accurate when the image is at
                the resolution the model was trained on, so pass that size when
                the file may differ.
        Returns:
            A 1-D numpy array of length message_length with {0, 1} integer bits.
        """
        # Load as an (H, W, C) float array in [0, 1], matching the training pipeline.
        image = Image.open(path).convert("RGB")
        if size is not None:
            image = image.resize((size, size))
        image = np.asarray(image, dtype=np.float32) / 255.0

        # To a (1, C, H, W) tensor on the model's device.
        image_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Run the decoder in eval mode (so BatchNorm uses running stats) without
        # tracking gradients, then hard-threshold the per-bit logits at 0 exactly
        # as training does.
        self.secret_decoder.eval()
        with torch.no_grad():
            logits = self.decode_batch(image_t)
        predicted_bits = (logits > 0).long().squeeze(0)

        return predicted_bits.cpu().numpy()

    def evaluate_noise_robustness(
        self,
        cover: np.ndarray,
        message: np.ndarray,
        noise_types: dict[str, list[int] | list[float] | None],
        save_folder: str | Path,
    ) -> None:
        """
        Watermarks ``cover`` with ``message``, then applies each requested noise type
        to the watermarked image independently and saves a titled plot for each. Also
        saves the original cover and the clean watermarked image.

        For every (noise type, parameter) the watermarked image is corrupted by just
        that noise, the message is decoded from the corrupted image, and the resulting
        bit accuracy (fraction of correctly recovered bits) is reported in the plot
        title.

        ``encode_image`` (not ``encode_batch``) is used to produce the watermarked
        image, so it runs under inference_mode and is detached to numpy; no autograd
        graph is retained for what may be a large image.

        Args:
            cover: an (H, W, C) [0, 1] image to watermark, as passed to encode_image.
            message: a (message_length,) array of {0, 1} bits, as passed to
                encode_image.
            noise_types: maps a noise type name (a key of the noiser's
                ``named_noise_functions``, e.g. "identity", "differentiable",
                "jpeg_compression", "crop", "rotate") to the parameters to plot. If the
                value is None, the noise is applied once with whatever parameter it
                samples on its own; otherwise one plot is produced per entry in the
                list. The list entries are interpreted per noise type: for the
                imagenet-c corruptions they are integer severities, and for "rotate"
                they are rotation angles in degrees (floats). Other noise types have no
                tunable parameter, so only None is valid for them.
            save_folder: directory the plots are written to (created if needed).
        """
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        # Watermark the cover. encode_image runs under inference_mode and returns a
        # detached numpy array, so no computational graph is kept around for the
        # (possibly large) image.
        stego = self.encode_image(cover, message)

        image_plotting.save_image_plot(cover, "cover", save_folder / "cover.png")
        image_plotting.save_image_plot(
            stego, "watermarked", save_folder / "watermarked.png"
        )

        named_noise_functions = self.noiser.named_noise_functions()
        message_bits = message.reshape(-1).astype(int)

        for noise_type, params in noise_types.items():
            if params is None:
                # No fixed parameter requested: apply the noise with whatever
                # parameter it samples on its own (random for imagenet-c / rotate,
                # n/a otherwise).
                self._noise_decode_and_plot(
                    stego,
                    named_noise_functions[noise_type],
                    label=noise_type,
                    filename=f"{noise_type}.png",
                    message_bits=message_bits,
                    save_folder=save_folder,
                )
            elif noise_type == NOISE_ROTATE:
                # For rotation the parameters are fixed angles (in degrees).
                for angle in params:
                    noise_func = self.noiser.rotate_function_at_angle(angle)
                    self._noise_decode_and_plot(
                        stego,
                        noise_func,
                        label=f"{noise_type} (angle {angle})",
                        filename=f"{noise_type}_angle_{angle}.png",
                        message_bits=message_bits,
                        save_folder=save_folder,
                    )
            else:
                # For the imagenet-c corruptions the parameters are severities.
                for severity in params:
                    noise_func = self.noiser.noise_function_at_severity(
                        noise_type, severity
                    )
                    self._noise_decode_and_plot(
                        stego,
                        noise_func,
                        label=f"{noise_type} (severity {severity})",
                        filename=f"{noise_type}_severity_{severity}.png",
                        message_bits=message_bits,
                        save_folder=save_folder,
                    )

    def _noise_decode_and_plot(
        self,
        stego: np.ndarray,
        noise_func,
        label: str,
        filename: str,
        message_bits: np.ndarray,
        save_folder: Path,
    ) -> None:
        """
        Applies ``noise_func`` to the watermarked image ``stego``, decodes the
        message, and saves a titled plot of the noised image whose title is ``label``
        plus the recovered bit accuracy.
        """
        # Apply the noise to the watermarked image off the autograd graph.
        stego_t = (
            torch.from_numpy(stego).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            noised_t = noise_func(stego_t)
        noised = noised_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        recovered = self.decode_image(np.expand_dims(noised, axis=0))
        bit_accuracy = float((recovered.reshape(-1).astype(int) == message_bits).mean())

        image_plotting.save_image_plot(
            noised,
            f"{label} (bit accuracy: {bit_accuracy:.2f})",
            save_folder / filename,
        )

    def train(self) -> None:
        """
        Trains the message encoder and secret decoder via curriculum learning,
        closely mirroring :meth:`RoSteALS.train`. The one structural difference is
        that cropping robustness is trained from the very first checkpoint (the
        whole purpose of StegoPatch), while every other noise type is withheld
        until the final checkpoint. The frozen image autoencoder is used as-is;
        only the message encoder and secret decoder are optimized.
        """
        # If a rolling auto-resume checkpoint is present, a previous run was
        # interrupted (e.g. Modal restarted it), so resume from it instead of
        # starting over.
        if os.path.exists(self.autosave_path):
            phase = self._autosave_phase()
            if phase >= 1:
                # Phases 1-4 map directly onto restart_training's curriculum, so
                # hand off and let it run from there to the end.
                self.restart_training(self.autosave_path, phase)
                return
            # Phase 0 is the tiny initial "baby" phase, which restart_training
            # doesn't cover. Restore the saved weights and fall through to rerun
            # train from the top; the baby set recovers in a few steps.
            self.load_model(self.autosave_path)

        assert len(self.dataset) == self.training_data_size

        self.secret_decoder.train()
        self.message_encoder.train()

        # Timestamp of when training started, used to name the saved weights.
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Crop with probability 0.5 until we are accurate enough to begin noising.
        self.noiser.set_probabilities(
            p_identity=1 - self.crop_probability,
            p_differentiable=0,
            p_imagenet={},
            p_crop=self.crop_probability,
            p_rotate=0
        )

        # =================== Checkpoint 0 begins =================
        # Train one or two mini batches of data until the 0.9 bit accuracy threshold is reached.
        # The baby_dataset contains only a couple minibatches worth of images so the max_epochs
        # here is large because we want to do many passes over those images.
        baby_indices = np.random.choice(
            len(self.dataset),
            size=min(self.first_exposure_set_size, len(self.dataset)),
            replace=False
        )
        baby_dataset = Subset(self.dataset, baby_indices.tolist())
        self.train_until(
            baby_dataset,
            bit_accuracy_threshold=0.9,
            max_epochs=self.num_epochs_for_small_batch,
            start_time=start_time,
            save_interval_steps=10_000,
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
            save_interval_epochs=1,
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
            save_interval_epochs=1,
            start_time=start_time,
            checkpoint=2
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint3.pt")

        # =================== Checkpoint 3 begins =================
        # Expose the model to the full training set and train until bit accuracy reaches 0.98.
        self.train_until(
            self.dataset,
            bit_accuracy_threshold=0.98,
            save_interval_epochs=1,
            start_time=start_time,
            checkpoint=3,
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint4.pt")

        # =================== Checkpoint 4 begins =================
        # Turn on the full noise blend (differentiable + imagenet-c corruptions, alongside
        # cropping) to finish training for robustness. The imagenet budget is split per
        # corruption via self.imagenet_probabilities; identity absorbs whatever is left
        # so the four scalars plus the per-corruption probabilities sum to 1.
        p_diff = p_crop = p_rotate = 0.225
        self.noiser.set_probabilities(
            p_identity=1.0 - (p_diff + p_crop + p_rotate + sum(self.imagenet_probabilities.values())),
            p_differentiable=p_diff,
            p_imagenet=self.imagenet_probabilities,
            p_crop=p_crop,
            p_rotate=p_rotate
        )
        self.train_until(
            self.dataset,
            max_epochs=self.num_epochs,
            save_interval_epochs=1,
            start_time=start_time,
            checkpoint=4
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint5.pt")

        # Training finished cleanly: drop the rolling auto-resume checkpoint so
        # the next fresh run isn't mistaken for an interrupted one.
        self._clear_autosave()

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
                4 - turn on the full noise blend.
                5 - focus training on just jpeg images.
        """
        # If a rolling auto-resume checkpoint is present it reflects more recent
        # progress than the explicitly requested checkpoint (e.g. the run was
        # restarted partway through this resumed training), so prefer it.
        if os.path.exists(self.autosave_path):
            autosave_phase = self._autosave_phase()
            if autosave_phase >= 1:
                save_path = self.autosave_path
                checkpoint = autosave_phase

        assert checkpoint in (1, 2, 3, 4, 5)

        # Restore weights, optimizer state, and the beta schedule progress.
        self.load_model(save_path)

        self.secret_decoder.train()
        self.message_encoder.train()

        # Timestamp of when this resumed run started, used to name new checkpoints.
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Cropping is enabled from the very first checkpoint, so every phase we can
        # resume into (1-4) trains against crops. The final phase widens this to the
        # full blend.
        self.noiser.set_probabilities(
            p_identity=1 - self.crop_probability,
            p_differentiable=0,
            p_imagenet={},
            p_crop=self.crop_probability,
            p_rotate=0
        )

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
                save_interval_epochs=1,
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
                save_interval_epochs=1,
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
                save_interval_epochs=1,
                start_time=start_time,
                checkpoint=3,
            )
            self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint4.pt")

        if checkpoint in (1, 2, 3, 4):
            # =================== Checkpoint 4 begins =================
            # Turn on the full noise blend (differentiable + imagenet-c, alongside
            # cropping) to finish training for robustness. The imagenet budget is split
            # per corruption via self.imagenet_probabilities; identity absorbs the rest.
            self.beta = self.beta_max
            p_diff = p_crop = p_rotate = 0.225
            self.noiser.set_probabilities(
                p_identity=1.0 - (p_diff + p_crop + p_rotate + sum(self.imagenet_probabilities.values())),
                p_differentiable=p_diff,
                p_imagenet=self.imagenet_probabilities,
                p_crop=p_crop,
                p_rotate=p_rotate
            )
            self.train_until(
                self.dataset,
                max_epochs=self.num_epochs,
                progress_bar="step",
                save_interval_epochs=1,
                start_time=start_time,
                checkpoint=4,
            )
            self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint5.pt")

        # ================= Checkpoint 5 begins ===============
        # Focus training on jpeg robustness: sample jpeg compression heavily while
        # keeping a little of the other branches so they don't regress. Splitting the
        # imagenet probability per corruption is exactly what makes this possible.
        self.update_flag = True
        self.noiser.set_probabilities(
            p_identity=0.1,
            p_differentiable=0.1,
            p_imagenet={NOISE_JPEG_COMPRESSION: 0.6},
            p_crop=0.1,
            p_rotate=0.1,
        )
        self.noiser.set_severity_range([1, 3])

        self.train_until(
            self.dataset,
            max_epochs=self.num_epochs,
            progress_bar="step",
            save_interval_epochs=1,
            start_time=start_time,
            checkpoint=5,
        )
        self.save_model(f"{self.models_dir}/stegopatch_{start_time}/checkpoint6.pt")

        # Training finished cleanly: drop the rolling auto-resume checkpoint so
        # the next fresh run isn't mistaken for an interrupted one.
        self._clear_autosave()

    def save_model(self, path: str, phase: int | None = None) -> None:
        """
        Saves the trainable network weights (message encoder and secret decoder)
        to ``path``. The frozen autoencoder is not saved.

        ``phase`` is only set for the rolling auto-resume checkpoint, where it
        records which curriculum phase produced the save so training can pick
        back up at the right place. It is left out entirely for the named /
        per-epoch checkpoints, so their on-disk format is unchanged.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
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
        }
        if phase is not None:
            state["phase"] = phase
        torch.save(state, path)

    def _save_autosave(self, phase: int) -> None:
        """
        Overwrites the single rolling auto-resume checkpoint, recording the
        curriculum ``phase`` that produced it. Writes to a temporary file and
        atomically renames it into place so a crash mid-write can never corrupt
        the existing checkpoint.
        """
        os.makedirs(self.autosave_dir, exist_ok=True)
        tmp_path = f"{self.autosave_path}.tmp"
        self.save_model(tmp_path, phase=phase)
        os.replace(tmp_path, self.autosave_path)

    def _autosave_phase(self) -> int:
        """
        Returns the curriculum phase recorded in the rolling auto-resume
        checkpoint (assumes it exists).
        """
        return torch.load(self.autosave_path, map_location=self.device)["phase"]

    def _clear_autosave(self) -> None:
        """Removes the rolling auto-resume checkpoint, if present."""
        if os.path.exists(self.autosave_path):
            os.remove(self.autosave_path)

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
            save_interval_steps=None,
            save_interval_epochs=None,
            progress_bar="epoch",
            start_time=None,
            checkpoint=None,
        ) -> None:
        """
        Trains the message encoder and secret decoder on the passed in dataset
        until the threshold is reached or max_epochs is reached.

        ``save_interval_steps`` and ``save_interval_epochs`` control how often the
        model is checkpointed during training. If ``save_interval_steps`` is set, the
        model is saved every that many steps; if ``save_interval_epochs`` is set, the
        model is saved every that many epochs. Either (or both) may be None, in which
        case no saving happens at that cadence. When any saving is requested,
        ``start_time`` and ``checkpoint`` must be provided so the per-interval
        checkpoints are written next to the ``train`` checkpoints (``{models_dir}/
        stegopatch_{start_time}/checkpoint{checkpoint}_step_{n}.pt`` or
        ``...checkpoint{checkpoint}_epoch_{n}.pt``); they share the same run directory
        and are grouped by the checkpoint phase that produced them. ``checkpoint`` is
        the number of the named checkpoint this training phase culminates in.

        ``progress_bar`` controls where the tqdm bar lives: "epoch" (default) wraps
        the outer epoch loop, "step" wraps the inner per-batch loop within each epoch.
        """
        assert progress_bar in ("epoch", "step")

        # Any per-interval checkpoint must be explicitly located: require a
        # start_time (and checkpoint) so we always know where they land.
        if save_interval_steps is not None or save_interval_epochs is not None:
            assert start_time is not None and checkpoint is not None
        max_epochs = max_epochs if max_epochs is not None else self.num_epochs
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Rolling buffer of the previous 18 bit accuracies, used to smooth out
        # noise before deciding whether the threshold has been reached.
        recent_bit_accuracies = deque(maxlen=18)

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

                # Apply noise only to what the decoder sees. Which corruptions are in
                # play is controlled entirely by the noiser's sampling probabilities,
                # which the train method adjusts as the curriculum progresses.
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

                # Rolling auto-resume checkpoint: overwrite a single file every
                # num_steps_to_save steps so an interrupted run can pick back up.
                # Independent of the named / per-interval checkpoints below; we
                # need the curriculum phase (``checkpoint``) to know where to
                # resume, so this only fires when a phase is supplied.
                if (
                    self.num_steps_to_save
                    and checkpoint is not None
                    and self.step % self.num_steps_to_save == 0
                ):
                    self._save_autosave(checkpoint)

                if save_interval_steps is not None and self.step % save_interval_steps == 0:
                    # Land per-step checkpoints in the same run directory the
                    # train named checkpoints use, prefixed with the checkpoint
                    # phase and tagged with the global step they correspond to.
                    self.save_model(
                        f"{self.models_dir}/stegopatch_{start_time}/"
                        f"checkpoint{checkpoint}_step_{self.step}.pt"
                    )

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

            if save_interval_epochs is not None and (epoch + 1) % save_interval_epochs == 0:
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

    def validate(
        self,
        progress_path: str | None = None,
        on_checkpoint=None,
        checkpoint_every: int = 10,
    ) -> dict:
        """
        Measures bit accuracy on the held-out test set, reporting it per individual
        noise type.

        Checkpointing: when ``progress_path`` is given, each batch's metrics are
        appended as one JSON line to that file as soon as they are computed, and
        ``on_checkpoint`` (if given) is called every ``checkpoint_every`` batches to
        flush the file to durable storage (e.g. a Modal volume commit). If the file
        already holds records from a previous run, those batches are skipped and
        validation resumes where it left off — the loader is deterministic
        (``shuffle=False``), so batch *i* is always the same images. A run that
        times out therefore loses at most ``checkpoint_every`` batches of work.

        Args:
            progress_path: Path to an append-only JSONL file used to persist and
                resume per-batch metrics. ``None`` disables checkpointing.
            on_checkpoint: Zero-arg callable invoked after a batch is flushed, every
                ``checkpoint_every`` batches and once more at the end, to persist the
                file durably. ``None`` disables durable flushing.
            checkpoint_every: How many batches between ``on_checkpoint`` calls.

        Returns:
            A dict mapping ``"quality_loss"`` and ``"bit_accuracy/{noise_name}"``
            (one entry per noise type) to their mean over the test set.
        """
        assert self.test_set is not None
        assert not self.log_tensorboard

        self.message_encoder.eval()
        self.secret_decoder.eval()

        # One bit-accuracy metric per individual noise type (identity, the
        # differentiable chain, each imagenet-c corruption, plus the StegoPatch
        # crop and rotate branches), so robustness can be read off per corruption
        # instead of against a random blend.
        noise_functions = self.noiser.named_noise_functions()
        metric_keys = ["quality_loss"] + [f"bit_accuracy/{name}" for name in noise_functions]

        # Cap validation to the first validation_set_size images of the test set
        # so we don't pay for the full ~40k every time.
        test_set = self.test_set
        if self.validation_set_size is not None:
            test_set = Subset(test_set, range(min(self.validation_set_size, len(test_set))))

        loader = DataLoader(test_set, self.batch_size, shuffle=False)
        num_steps = len(loader)

        # Reload any per-batch records left by a previous (e.g. timed-out) run and
        # resume after the last one. Records are assumed contiguous from batch 0.
        records = _read_progress_records(progress_path) if progress_path else []
        start_batch = len(records)
        if start_batch >= num_steps:
            return _aggregate_records(records, metric_keys)
        if start_batch:
            print(f"Resuming validation from batch {start_batch}/{num_steps}", flush=True)

        progress_file = open(progress_path, "a", encoding="utf-8") if progress_path else None
        try:
            with torch.no_grad():
                for i, covers in enumerate(tqdm(loader, desc="steps")):
                    if i < start_batch:
                        continue
                    covers = covers.to(self.device)
                    messages = torch.randint(
                        0, 2, (covers.shape[0], self.message_length), device=self.device
                    ).float()
                    stego_images = self.encode_batch(covers, messages)

                    record = {
                        "batch": i,
                        "quality_loss": self.get_quality_loss(covers, stego_images).item(),
                    }
                    # Pass the same stego batch through each noise type independently.
                    for name, noise_func in noise_functions.items():
                        recovered_messages = self.decode_batch(noise_func(stego_images))
                        predicted_bits = (recovered_messages > 0).float()
                        record[f"bit_accuracy/{name}"] = (
                            (predicted_bits == messages).float().mean().item()
                        )

                    records.append(record)
                    if progress_file is not None:
                        progress_file.write(json.dumps(record) + "\n")
                        progress_file.flush()
                        if on_checkpoint is not None and (i + 1) % checkpoint_every == 0:
                            on_checkpoint()
        finally:
            if progress_file is not None:
                progress_file.close()
                if on_checkpoint is not None:
                    on_checkpoint()  # flush the tail batches written since the last commit

        return _aggregate_records(records, metric_keys)


def _read_progress_records(progress_path: str) -> list[dict]:
    """Read per-batch metric records previously appended to ``progress_path``.

    Returns an empty list if the file does not exist yet. A trailing line that
    fails to parse (a torn write from a process killed mid-flush) is dropped, so
    resume picks up from the last fully-written batch.
    """
    if not os.path.exists(progress_path):
        return []

    records = []
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # torn final line; everything before it is intact
    return records


def _aggregate_records(records: list[dict], metric_keys: list[str]) -> dict:
    """Average each metric across the per-batch ``records`` into the final results dict."""
    results = {key: 0.0 for key in metric_keys}
    if not records:
        return results
    for record in records:
        for key in metric_keys:
            results[key] += record[key]
    for key in metric_keys:
        results[key] /= len(records)
    return results
