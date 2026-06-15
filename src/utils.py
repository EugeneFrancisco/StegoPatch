"""
Utils file for odds and ends
"""
import lpips
import torch

# Lazily-built LPIPS network, cached so we only download/instantiate it once.
_lpips_net: lpips.LPIPS | None = None


def _get_lpips_net(device: torch.device) -> lpips.LPIPS:
    global _lpips_net
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
