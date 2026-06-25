# pylint: skip-file
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

import src.utils as utils
import src.plotting.image_plotting as image_plotting
from src.watermarkers.stegopatch import StegoPatch
from src.noisers.stegopatch_noiser import StegoPatchNoiser

DEBUGGING = 0
TRAINING = 1
TESTING = 2

MODE = TRAINING

DATA_DIR = Path("data/train2017")
# vq-f4 was trained on 256x256 crops, so we work at that resolution.
PATCH_SIZE = 96
IMAGE_SIZE = 384
CROP_SIZE = 288
MESSAGE_LENGTH = 20
if MODE in (TRAINING, TESTING):
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    NUM_EPOCHS_FOR_SMALL_BATCH = 50_000
    FIRST_EXPOSURE_SIZE = 8
    SECOND_EXPOSURE_SIZE = 50_000
    TRAINING_DATA_SIZE = 100_000
else:
    BATCH_SIZE = 2
    NUM_EPOCHS = 2
    NUM_EPOCHS_FOR_SMALL_BATCH = 2
    FIRST_EXPOSURE_SIZE = 8
    SECOND_EXPOSURE_SIZE = 16
    TRAINING_DATA_SIZE = 32

if MODE == TRAINING:
    LOG_TENSORBOARD = True
else:
    LOG_TENSORBOARD = False

C_IMAGE = 3
H_LITTLE = PATCH_SIZE / 8
W_LITTLE = PATCH_SIZE / 8
C_LITTLE = 3
ALPHA = 1.5
BETA_MIN = 0.08
BETA_MAX = 10
BETA_DELTA = (BETA_MAX - BETA_MIN) / 5_000 # just from observation, it seems like 5k steps till convergence roughly
LEARNING_RATE = 2e-5

def get_default_device() -> str:
    """Picks the best available torch device (cuda on Modal, mps locally)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_configs(
    data_path: Path,
    device: str | None,
    models_dir: str,
    tensorboard_log_dir: str,
    test_set_path: Path | None = None
) -> dict:
    """Builds the standard RoSteALS training configs and dataset.

    Args:
        data_path: Path to the .npy file holding the training images.
        device: Torch device to run on (e.g. "cuda" or "cpu"); if None, the
            default device is auto-selected.
        models_dir: Directory where model checkpoints are saved and loaded.
        tensorboard_log_dir: Directory where TensorBoard logs are written.
        test_set_path: the path to the .npy file folding testing images.
    """
    device = device or get_default_device()
    dataset = utils.NpyImageDataset(data_path)
    dataset = Subset(dataset, range(TRAINING_DATA_SIZE))
    test_set = utils.NpyImageDataset(test_set_path) if test_set_path is not None else None
    noiser_configs = {
        "p_differentiable": 0,
        "p_imagenet": 0,
        "p_crop": 0.5,
        "p_identity": 0.5,
        "p_rotate": 0,
        "w_image": IMAGE_SIZE,
        "h_image": IMAGE_SIZE,
        "crop_size": CROP_SIZE,
        "rotation_lower_bound": -30,
        "rotation_upper_bound": 30,
    }
    noiser = StegoPatchNoiser(noiser_configs)
    return {
        "device": device,
        "autoencoder_type": "VQGAN",
        "message_length": MESSAGE_LENGTH,
        "patch_size": PATCH_SIZE,
        "c_image": C_IMAGE,
        "h_little": H_LITTLE,
        "w_little": W_LITTLE,
        "c_little": C_LITTLE,
        "alpha": ALPHA,
        "beta_min": BETA_MIN,
        "beta_max": BETA_MAX,
        "beta_delta": BETA_DELTA,
        "learning_rate": LEARNING_RATE,
        "dataset": dataset,
        "test_set": test_set,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "num_epochs_for_small_batch": NUM_EPOCHS_FOR_SMALL_BATCH,
        "training_data_sizes": {
            0: FIRST_EXPOSURE_SIZE,
            1: SECOND_EXPOSURE_SIZE,
            2: TRAINING_DATA_SIZE
        },
        "models_dir": models_dir,
        "tensorboard_log_dir": tensorboard_log_dir,
        "log_tensorboard": LOG_TENSORBOARD,
        "noiser": noiser
    }

def _build_stegopatch(
    data_path: Path,
    device: str | None,
    models_dir: str,
    tensorboard_log_dir: str,
    test_set_path: Path | None = None
) -> StegoPatch:
    """Builds a RoSteALSPatcher with the standard training configs and dataset.

    Args:
        data_path: Path to the .npy file holding the training images.
        device: Torch device to run on (e.g. "cuda" or "cpu"); if None, the
            default device is auto-selected.
        models_dir: Directory where model checkpoints are saved and loaded.
        tensorboard_log_dir: Directory where TensorBoard logs are written.
    """
    configs = build_configs(data_path, device, models_dir, tensorboard_log_dir, test_set_path)
    return StegoPatch(configs)


def main(
    data_path: Path,
    device: str | None,
    models_dir: str,
    tensorboard_log_dir: str,
):
    stegopatch = _build_stegopatch(
        data_path,
        device,
        models_dir,
        tensorboard_log_dir
    )
    stegopatch.train()



def restart(
    save_path: str,
    checkpoint: int,
    data_path: Path,
    device: str | None,
    models_dir: str,
    tensorboard_log_dir: str,
):
    """Resumes training from the checkpoint .pt file at ``save_path``."""
    stegopatch = _build_stegopatch(data_path, device, models_dir, tensorboard_log_dir)
    stegopatch.restart_training(save_path, checkpoint)



if __name__ == "__main__":
    main(
        data_path=Path("data/train2017_numpy_384.npy"),
        device=None,
        models_dir="results/stegopatch/debugging1/models",
        tensorboard_log_dir="results/stegopatch/debugging1/runs",
    )
