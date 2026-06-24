"""
Utils file for odds and ends
"""
import random
from functools import lru_cache
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class NpyImageDataset(Dataset):
    """
    Wraps a precomputed (N, C, H, W) uint8 .npy array (see precompute_image_tensors)
    as a PyTorch Dataset. The file is memory-mapped, so images are read from disk
    on demand rather than loaded into RAM up front.
    """

    def __init__(self, npy_path: Path):
        # mmap_mode="r" keeps the array on disk; slices are read lazily per __getitem__.
        self.images = np.load(npy_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Copy the mmap slice into RAM, then convert uint8 [0, 255] -> float [0, 1].
        image = np.asarray(self.images[index], dtype=np.float32) / 255.0
        return torch.from_numpy(image)


@lru_cache(maxsize=None)
def list_images(data_dir: Path) -> tuple[Path, ...]:
    """Scans data_dir for jpgs once and caches the result (scanning 118k files is slow)."""
    return tuple(data_dir.glob("*.jpg"))


def load_random_image(data_dir: Path, size: int | None = None) -> np.ndarray:
    """Loads a random image as an (H, W, C) float array in [0, 1].

    If ``size`` is given the image is resized to a ``size`` x ``size`` square;
    otherwise it is returned at its original resolution.
    """
    path = random.choice(list_images(data_dir))
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size))
    return np.asarray(image, dtype=np.float32) / 255.0


def load_random_images(data_dir: Path, size: int, batch_size: int) -> torch.Tensor:
    """Loads batch_size random images as a (batch_size, C, H, W) float tensor in [0, 1]."""
    images = [load_random_image(data_dir, size) for _ in range(batch_size)]
    batch = np.stack(images, axis=0)  # (batch_size, H, W, C)
    batch = np.transpose(batch, (0, 3, 1, 2))  # (batch_size, C, H, W)
    return torch.from_numpy(batch)


def load_all_images(data_dir: Path, size: int) -> torch.Tensor:
    """Loads every image in data_dir as an (N, C, H, W) float tensor in [0, 1]."""
    images = [
        np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32)
        / 255.0
        for path in list_images(data_dir)
    ]
    batch = np.stack(images, axis=0)  # (N, H, W, C)
    batch = np.transpose(batch, (0, 3, 1, 2))  # (N, C, H, W)
    return torch.from_numpy(batch)


def precompute_image_tensors(data_dir: Path, out_path: Path, size: int) -> None:
    """
    Precomputes every image in data_dir into a single (N, C, size, size) uint8 array
    saved to out_path, so a DataLoader can later memory-map it and skip the slow JPEG
    decode/resize.

    Pixels are stored as uint8 in [0, 255] (4x smaller than float32 and lossless);
    divide by 255 at load time to get a float tensor in [0, 1]. The array is written
    straight to disk via a memmap, so building it does not load every image into RAM.
    """
    paths = list_images(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Allocate the array on disk up front, then fill it one image at a time.
    array = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint8, shape=(len(paths), 3, size, size)
    )
    for i, path in enumerate(tqdm(paths)):
        image = Image.open(path).convert("RGB")
        # Center-crop the longer side away so the image is square (preserving aspect
        # ratio), then downsample to the target resolution without further cropping.
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize((size, size))
        array[i] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)  # (C, size, size)
    array.flush()


# Lazily-built LPIPS network, cached so we only download/instantiate it once.
_lpips_net: lpips.LPIPS | None = None


def _get_lpips_net(device: torch.device) -> lpips.LPIPS:
    global _lpips_net # pylint: disable=global-statement
    if _lpips_net is None:
        # AlexNet backbone is the configuration recommended as a perceptual loss.
        _lpips_net = lpips.LPIPS(net="alex")
        # Freeze the backbone: we only want gradients w.r.t. the images, not the net.
        _lpips_net.eval()
        for param in _lpips_net.parameters():
            param.requires_grad_(False)
    return _lpips_net.to(device)

def rgb_to_yuv(images: torch.Tensor) -> torch.Tensor:
    """
    Converts the passed in images from RGB into YUV space.
    Args:
        images: a (B, C, H, W) tensor of images where the C dimension is in RGB.
    Returns:
        A (B, C, H, W) tensor of images where the C dimension is in YUV.
    """
    # BT.601 RGB -> YUV conversion matrix (rows map RGB to Y, U, V).
    weight = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001],
        ],
        dtype=images.dtype,
        device=images.device,
    )
    # einsum keeps the op differentiable: gradients flow back through `images`.
    return torch.einsum("oc,bchw->bohw", weight, images)

def lpips_loss(originals: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
    """
    Calculates the LPIP loss between input and target, where originals are the original
    cover images and modified are the stego images after watermarking.
    Args:
        originals: a (B, C, H, W) tensor of cover images in [0, 1].
        modified: a (B, C, H, W) tensor of stego images in [0, 1].
    Returns:
        A scalar tensor: the mean LPIPS distance over the batch. The graph is kept
        intact so gradients flow back to ``modified`` (and ``originals``).
    """
    net = _get_lpips_net(modified.device)
    # LPIPS expects inputs in [-1, 1]; the affine map is differentiable.
    originals = originals * 2.0 - 1.0
    modified = modified * 2.0 - 1.0
    # Returns a (B, 1, 1, 1) tensor of per-image distances; average to a scalar.
    return net(originals, modified).mean()

def bce_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the binary cross entropy loss between the input and the target.
    Args:
        input: a (B, message_length) tensor of raw logits from the decoder.
        target: a (B, message_length) tensor of {0, 1} target message bits.
    Returns:
        The binary cross entropy between the two tensors, averaged across
        batches.
    """
    # `with_logits` applies the sigmoid internally for numerical stability, and
    # the op is differentiable so gradients flow back to `input` (the decoder).
    return torch.nn.functional.binary_cross_entropy_with_logits(
        input, target.to(input.dtype)
    )
