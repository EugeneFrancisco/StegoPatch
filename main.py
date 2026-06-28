# pylint: skip-file
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

import src.utils as utils
import src.plotting.image_plotting as image_plotting
from src.watermarkers.stegopatch import StegoPatch
from src.watermarkers.stegopatch_legacy import StegoPatchLegacy
from src.noisers.stegopatch_noiser import (
    StegoPatchNoiser,
    NOISE_JPEG_COMPRESSION,
    NOISE_CROP,
    NOISE_ROTATE,
    NOISE_DIFFERENTIABLE,
)

DEBUGGING = 0
TRAINING = 1
TESTING = 2

MODE = TESTING

DATA_DIR = Path("data/train2017")
# vq-f4 was trained on 256x256 crops, so we work at that resolution.
PATCH_SIZE = 96
IMAGE_SIZE = 384
CROP_SIZE = 288
MESSAGE_LENGTH = 20
# The full test set has ~40k images; validating on the first 20k is plenty and
# roughly halves validation time.
VALIDATION_SET_SIZE = 20_000
P_CROP = 0.25
P_ROTATE = 0.25
P_IDENTITY = 0.5
ROTATION_BOUND = 30
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
BETA_MIN = 11
BETA_MAX = 15
BETA_DELTA = (BETA_MAX - BETA_MIN) / 4_500
LEARNING_RATE = 2e-5
# How often (in steps) to overwrite the rolling auto-resume checkpoint so a run
# that gets restarted (e.g. by Modal) can pick back up where it left off.
NUM_STEPS_TO_SAVE = 5_000

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
        # No imagenet corruptions at construction; the curriculum sets the real
        # per-corruption probabilities. This is a {corruption_name: probability} dict.
        "p_imagenet": {},
        "p_crop": P_CROP,
        "p_identity": P_IDENTITY,
        "p_rotate": P_ROTATE,
        "w_image": IMAGE_SIZE,
        "h_image": IMAGE_SIZE,
        "crop_size": CROP_SIZE,
        "rotation_lower_bound": -ROTATION_BOUND,
        "rotation_upper_bound": ROTATION_BOUND,
    }
    noiser = StegoPatchNoiser(noiser_configs)

    # Per-corruption imagenet-c sampling probabilities for the final training blend.
    # Start from an equal split of the 0.225 imagenet budget across every corruption
    # (the historical behaviour), then override individual corruptions to weight, say,
    # jpeg over gaussian blur. Keep the dict's total at 0.225 so the overall blend
    # still sums to 1, e.g.:
    #   imagenet_probabilities[NOISE_JPEG_COMPRESSION] = 0.1
    imagenet_probabilities = noiser.uniform_imagenet_probabilities(0.225)
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
        "validation_set_size": VALIDATION_SET_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "num_epochs_for_small_batch": NUM_EPOCHS_FOR_SMALL_BATCH,
        "training_data_sizes": {
            0: FIRST_EXPOSURE_SIZE,
            1: SECOND_EXPOSURE_SIZE,
            2: TRAINING_DATA_SIZE
        },
        "models_dir": models_dir,
        # The rolling auto-resume checkpoint lives in its own directory, distinct
        # from the run-named checkpoints under models_dir.
        "autosave_dir": f"{models_dir}/autosave",
        "num_steps_to_save": NUM_STEPS_TO_SAVE,
        "tensorboard_log_dir": tensorboard_log_dir,
        "log_tensorboard": LOG_TENSORBOARD,
        "noiser": noiser,
        "imagenet_probabilities": imagenet_probabilities,
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
    stegopatch = _build_stegopatch(data_path, device, models_dir, tensorboard_log_dir)
    stegopatch.load_model("results/stegopatch/experiment_2/models/stegopatch_2026-06-27_15-57-27/checkpoint4_epoch_2.pt")
    cover = image_plotting.load_image(
        Path("data/sample_images/cat/000000236877.png"),
        IMAGE_SIZE + PATCH_SIZE + PATCH_SIZE,
        IMAGE_SIZE + PATCH_SIZE + PATCH_SIZE,
    )
    message = np.empty(MESSAGE_LENGTH)
    for i in range(MESSAGE_LENGTH):
        if i % 2:
            message[i] = 0
        else:
            message[i] = 1

    stegopatch.evaluate_noise_robustness(
        cover,
        message,
        {
            NOISE_JPEG_COMPRESSION: [1, 2, 3],
            NOISE_CROP: None,
            NOISE_ROTATE: [30.0],
            NOISE_DIFFERENTIABLE: None,
        },
        save_folder=Path("data/sample_images/cat/noise_robustness"),
    )
    

    
    



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
        data_path=Path("data/numpy/train2017_numpy_384.npy"),
        device=None,
        models_dir=Path("results/stegopatch/debugging1/models"),
        tensorboard_log_dir=Path("results/stegopatch/debugging1/runs"),
    )
