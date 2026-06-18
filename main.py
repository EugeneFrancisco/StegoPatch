# pylint: skip-file
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Subset
from typing import Optional

import src.utils as utils
from src.watermarkers.rosteals import RoSteALS

DATA_DIR = Path("data/train2017")
# vq-f4 was trained on 256x256 crops, so we work at that resolution.
IMAGE_SIZE = 256
MESSAGE_LENGTH = 50
BATCH_SIZE = 8

C_IMAGE = 3
H_IMAGE = IMAGE_SIZE
W_IMAGE = IMAGE_SIZE
H_LITTLE = IMAGE_SIZE / 8
W_LITTLE = IMAGE_SIZE / 8
C_LITTLE = 3
ALPHA = 1.5
BETA_MIN = 0.1
BETA_MAX = 10
BETA_DELTA = (BETA_MAX - BETA_MIN) / 5_000 # just from observation, it seems like 5k steps till convergence roughly
NUM_EPOCHS_FOR_LARGE_BATCH = 8
NUM_EPOCHS_FOR_SMALL_BATCH = 200_000
LEARNING_RATE = 2e-5
TRAINING_SUBSET_SIZE = 8
TRAINING_DATA_SIZE = 50_000
LOG_TENSORBOARD = False


def get_default_device() -> str:
    """Picks the best available torch device (cuda on Modal, mps locally)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_rosteals(
    data_path: Path,
    device: str | None,
    models_dir: str,
    tensorboard_log_dir: str,
) -> RoSteALS:
    """Builds a RoSteALS with the standard training configs and dataset."""
    device = device or get_default_device()
    dataset = utils.NpyImageDataset(data_path)
    dataset = Subset(dataset, range(TRAINING_DATA_SIZE))
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
        "dataset": dataset,
        "num_epochs": NUM_EPOCHS_FOR_LARGE_BATCH,
        "num_epochs_for_small_batch": NUM_EPOCHS_FOR_SMALL_BATCH,
        "batch_size": BATCH_SIZE,
        "training_subset_size": TRAINING_SUBSET_SIZE,
        "training_data_size": TRAINING_DATA_SIZE,
        "models_dir": models_dir,
        "tensorboard_log_dir": tensorboard_log_dir,
        "log_tensorboard": LOG_TENSORBOARD
    }
    return RoSteALS(configs)


def main(
    data_path: Path = Path("data/train2017_numpy_256.npy"),
    device: str | None = None,
    models_dir: str = "models",
    tensorboard_log_dir: str = "runs/rosteals",
):
    rosteals = _build_rosteals(data_path, device, models_dir, tensorboard_log_dir)
    rosteals.load_model("models/rosteals_2026-06-18_16-39-39/checkpoint3.pt")
    image = utils.load_random_image(DATA_DIR, IMAGE_SIZE)
    message = np.random.randint(0, 2, (MESSAGE_LENGTH, 1))
    stego_image = rosteals.encode_image(image, message)
    recovered_message = rosteals.decode_image(stego_image)

    bit_accuracy = np.sum(recovered_message == message)/MESSAGE_LENGTH

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[1].imshow(stego_image)
    axes[1].set_title(f"stego (bit accuracy {bit_accuracy:.2f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    out_path = Path("results/roundtrip.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def restart(
    save_path: str,
    checkpoint: int,
    data_path: Path = Path("data/train2017_numpy_256.npy"),
    device: str | None = None,
    models_dir: str = "models",
    tensorboard_log_dir: str = "runs/rosteals",
):
    """Resumes training from the checkpoint .pt file at ``save_path``."""
    rosteals = _build_rosteals(data_path, device, models_dir, tensorboard_log_dir)
    rosteals.restart_training(save_path, checkpoint)



if __name__ == "__main__":
    main()
