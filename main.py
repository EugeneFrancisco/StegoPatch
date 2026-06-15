# pylint: skip-file
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.autoencoders.vqgan import VQGAN
from src.watermarkers.rosteals import RoSteALS

DEVICE = "mps"

DATA_DIR = Path("data/train2017")
# vq-f4 was trained on 256x256 crops, so we work at that resolution.
IMAGE_SIZE = 256
MESSAGE_LENGTH = 100


def load_random_image(data_dir: Path, size: int) -> np.ndarray:
    """Loads a random image as an (H, W, C) float array in [0, 1], resized to a square."""
    path = random.choice(list(data_dir.glob("*.jpg")))
    print(f"Loaded {path}")
    image = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(image, dtype=np.float32) / 255.0


def main():
    configs = {
        "device": "mps",
        "autoencoder_type": "VQGAN",
        "message_length": MESSAGE_LENGTH,
        "c_image": 3,
        "h_image": IMAGE_SIZE,
        "w_image": IMAGE_SIZE,
        "h_little": IMAGE_SIZE/8,
        "w_little": IMAGE_SIZE/8,
        "c_little": 16,
    }
    rosteals = RoSteALS(configs)
    image = load_random_image(DATA_DIR, IMAGE_SIZE)
    message = np.random.randint(0, 2, (MESSAGE_LENGTH, 1))
    watermarked = rosteals.encode_image(image, message)

    # Save the original and reconstruction side by side so the round-trip is visible.
    side_by_side = np.concatenate([image, watermarked], axis=1)
    out = Image.fromarray((side_by_side * 255).round().astype(np.uint8))
    out.save("results/roundtrip.png")
    print("Wrote roundtrip.png (original | reconstruction)")


if __name__ == "__main__":
    main()
