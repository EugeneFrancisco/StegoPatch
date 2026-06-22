"""
Plot watermarked images from different checkpoints of the model.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.watermarkers.image_watermarker import ImageWatermarker
from src.utils import load_random_image
from main import IMAGE_SIZE, MESSAGE_LENGTH, _build_rosteals

# Directory where the cover/stego image plots are saved.
PLOT_SAVE_DIR = Path("results/experiment_1/plots")


def save_image_plot(image: np.ndarray, title: str, filename: str) -> None:
    """Saves a single (H, W, C) [0, 1] image as a titled plot in PLOT_SAVE_DIR."""
    PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(PLOT_SAVE_DIR / filename, bbox_inches="tight")
    plt.close(fig)

def main():
    # Dictionary where keys are the checkpoints that the models refer to and values are the 
    # paths to the model .pt files.
    paths = {
        "Warmup 1": "results/experiment_1/models/rosteals_2026-06-18_07-07-57/checkpoint1.pt",
        "Warmup 2": "results/experiment_1/models/rosteals_2026-06-18_15-40-02/checkpoint2.pt",
        "Exposure 1": "results/experiment_1/models/rosteals_2026-06-18_16-39-39/checkpoint3.pt",
        "Exposure 2": "results/experiment_1/models/rosteals_2026-06-18_19-27-00/checkpoint4.pt",
        "Noising": "results/experiment_1/models/rosteals_2026-06-21_02-20-52/checkpoint5_epoch_1.pt"
    }

    rosteals: ImageWatermarker = _build_rosteals(
        data_path="data/train2017_numpy_256.npy",
        device="mps",
        models_dir=None,
        tensorboard_log_dir=None
    )

    cover: np.ndarray = load_random_image(Path("data/train2017"), IMAGE_SIZE)
    message: np.ndarray = np.random.randint(0, 2, (MESSAGE_LENGTH, 1))

    save_image_plot(cover, "Original cover image", "cover.png")

    for checkpoint, path in paths.items():
        rosteals.load_model(path)
        stego: np.ndarray = rosteals.encode_image(cover, message)
        # Use a filesystem-safe version of the checkpoint name for the filename.
        slug = checkpoint.lower().replace(" ", "_")
        save_image_plot(stego, f"Stego image: {checkpoint}", f"stego_{slug}.png")


if __name__ == "__main__":
    main()
