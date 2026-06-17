# pylint: skip-file
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from PIL import Image

from src.autoencoders.vqgan import VQGAN
import src.utils as utils
from src.watermarkers.rosteals import RoSteALS

DATA_DIR = Path("data/train2017")
# vq-f4 was trained on 256x256 crops, so we work at that resolution.
IMAGE_SIZE = 256
MESSAGE_LENGTH = 50
BATCH_SIZE = 4

C_IMAGE = 3
H_IMAGE = IMAGE_SIZE
W_IMAGE = IMAGE_SIZE
H_LITTLE = IMAGE_SIZE / 8
W_LITTLE = IMAGE_SIZE / 8
C_LITTLE = 3
ALPHA = 1.5
BETA_MIN = 0.1
BETA_MAX = 10
BETA_DELTA = 1
NUM_EPOCHS_FOR_LARGE_BATCH = 20
NUM_EPOCHS_FOR_SMALL_BATCH = 200_000
LEARNING_RATE = 2e-5
TRAINING_SUBSET_SIZE = 4
TRAINING_DATA_SIZE = 50_000


def get_default_device() -> str:
    """Picks the best available torch device (cuda on Modal, mps locally)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(
    data_path: Path = Path("data/train2017_numpy_256.npy"),
    device: str | None = None,
    models_dir: str = "models",
    tensorboard_log_dir: str = "runs/rosteals",
):
    device = device or get_default_device()
    configs = {
        "device": device,
        "autoencoder_type": "VQGAN",
        "message_length": MESSAGE_LENGTH,
        "c_image": C_IMAGE,
        "h_image": H_IMAGE,
        "w_image": W_IMAGE,
        "h_little": H_LITTLE,
        "w_little": W_LITTLE,
        "c_little": C_LITTLE,
        "alpha": ALPHA,
        "beta_min": BETA_MIN,
        "beta_max": BETA_MAX,
        "beta_delta": BETA_DELTA,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS_FOR_LARGE_BATCH,
        "num_epochs_for_small_batch": NUM_EPOCHS_FOR_SMALL_BATCH,
        "batch_size": BATCH_SIZE,
        "training_subset_size": TRAINING_SUBSET_SIZE,
        "training_data_size": TRAINING_DATA_SIZE,
        "models_dir": models_dir,
        "tensorboard_log_dir": tensorboard_log_dir,
    }
    rosteals = RoSteALS(configs)
    dataset = utils.NpyImageDataset(data_path)
    dataset = Subset(dataset, range(TRAINING_DATA_SIZE))
    rosteals.train(dataset)
    


if __name__ == "__main__":
    main()
