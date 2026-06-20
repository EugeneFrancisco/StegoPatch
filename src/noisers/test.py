"""
Just some code to test the noisers
"""
# pylint: skip-file

from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
import src.utils as utils
from src.noisers.rosteals_noiser import RoSteALSNoiser

DATA_DIR = Path("data/train2017")
IMAGE_SIZE = 256

def main():
    image = utils.load_random_image(DATA_DIR, IMAGE_SIZE)
    image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)  # (1, C, H, W)
    configs = {
        "p_differentiable": 0,
        "p_imagenet": 1,
        "p_identity": 0,
        "w_image": IMAGE_SIZE,
        "h_image": IMAGE_SIZE
    }
    noiser = RoSteALSNoiser(configs)
    noise_type = noiser.sample_noise_type()
    print(noise_type)
    noise_func = noiser.get_noise_function(noise_type)
    image_prime = noise_func(image)

    def to_hwc(x):  # (1, C, H, W) tensor -> (H, W, C) numpy for imshow
        return x[0].detach().permute(1, 2, 0).numpy()

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(to_hwc(image))
    axes[0].set_title("image")
    axes[1].imshow(to_hwc(image_prime))
    axes[1].set_title("image_prime")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()