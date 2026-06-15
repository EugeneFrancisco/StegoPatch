# pylint: skip-file
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.autoencoders.vqgan import VQGAN
from src.utils import load_random_image, load_random_images
from src.watermarkers.rosteals import RoSteALS

DEVICE = "mps"

DATA_DIR = Path("data/train2017")
# vq-f4 was trained on 256x256 crops, so we work at that resolution.
IMAGE_SIZE = 256
MESSAGE_LENGTH = 100
BATCH_SIZE = 4


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
        "alpha": 1,
        "beta": 1,
    }
    rosteals = RoSteALS(configs)
    # image = load_random_image(DATA_DIR, IMAGE_SIZE)
    # message = np.random.randint(0, 2, (MESSAGE_LENGTH, 1))
    # watermarked = rosteals.encode_image(image, message)
    # recovered = rosteals.decode_image(watermarked)
    # # Save the original and reconstruction side by side so the round-trip is visible.
    # side_by_side = np.concatenate([image, watermarked], axis=1)
    # out = Image.fromarray((side_by_side * 255).round().astype(np.uint8))
    # out.save("results/roundtrip.png")
    # print("Wrote roundtrip.png (original | reconstruction)")

    covers = load_random_images(DATA_DIR, IMAGE_SIZE, BATCH_SIZE).to(configs["device"])
    messages = torch.from_numpy(
        np.random.randint(0, 2, (BATCH_SIZE, MESSAGE_LENGTH))
    ).float().to(configs["device"])
    stego_images = rosteals.encode_batch(covers, messages)
    recovered_messages = rosteals.decode_batch(stego_images)

    recovery_loss, quality_loss = rosteals.get_loss(
        covers,
        messages,
        stego_images,
        recovered_messages
    )

    import ipdb; ipdb.set_trace()
    


if __name__ == "__main__":
    main()
