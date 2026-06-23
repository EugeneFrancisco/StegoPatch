"""
This file defines the StegoPatch Watermarking class.
"""
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights

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

        # A flag for when to begin applying noise during training.
        self.begin_noising: bool = False

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

    def train(self):
        # ============ Checkpoint 0 =============
        # Train on only one minibatch of images until bit accuracy crosses 0.9.

        # ============ Checkpoint 1 =============
        # Reveal the model to much more of the data and train until the bit accuracy crosses 0.8

        # ============ Checkpoint 2 =============
        # Begin incrementing beta from beta_min to beta_max until bit accuracy reaches 0.95.

        # ============ Checkpoint 3 =============
        # Expose the full training set until bit accuracy reaches 0.98

        # ============ Checkpoint 4 =============
        # Add other forms of noise.
        pass

    def validate(self):
        pass
