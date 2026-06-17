"""
Modal entrypoint: trains the RoSteALS watermarker on a cloud GPU and exposes a
live TensorBoard dashboard over a Modal tunnel while it runs.

One-time setup (upload the ~23 GB precomputed training array to a Volume):

    modal volume put rosteals-data \\
        data/train2017_numpy_256.npy train2017_numpy_256.npy

Run training (prints a public TensorBoard URL near the top of the logs):

    modal run modal_app.py

Fetch the trained weights afterwards (timestamp is printed at the end of the run):

    modal volume get rosteals-output models/rosteals_<timestamp>/final.pt ./final.pt
"""
import subprocess
from pathlib import Path

import modal

APP_NAME = "rosteals-watermark"

# Holds the precomputed (N, 3, 256, 256) uint8 .npy training array.
data_volume = modal.Volume.from_name("rosteals-data", create_if_missing=True)
# Holds training outputs: model checkpoints and TensorBoard run logs.
output_volume = modal.Volume.from_name("rosteals-output", create_if_missing=True)

DATA_DIR = "/data"
OUTPUT_DIR = "/output"
DATA_FILE = "train2017_numpy_256.npy"


def _download_pretrained_weights():
    """Bake the frozen pretrained nets into the image so runs never re-download them."""
    from diffusers import VQModel
    from torchvision.models import resnet50, ResNet50_Weights
    import lpips

    VQModel.from_pretrained("xvjiarui/ldm-vq-f4")
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    lpips.LPIPS(net="alex")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "lpips",
        "numpy",
        "pillow",
        "tqdm",
        "tensorboard",
        "accelerate",
    )
    .run_function(_download_pretrained_weights)
    # Ship the local source last so editing it doesn't bust the build cache above.
    .add_local_python_source("src", "main")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="A10G",
    volumes={DATA_DIR: data_volume, OUTPUT_DIR: output_volume},
    timeout=24 * 60 * 60,
)
def train():
    import main as train_main

    # TensorBoard reads the same local directory the trainer writes to, so the
    # dashboard updates in real time (no Volume reload lag).
    subprocess.Popen(
        ["tensorboard", "--logdir", f"{OUTPUT_DIR}/runs",
         "--host", "0.0.0.0", "--port", "6006"]
    )

    with modal.forward(6006) as tunnel:
        print(f"\n>>> TensorBoard live at: {tunnel.url}\n", flush=True)
        try:
            train_main.main(
                data_path=Path(f"{DATA_DIR}/{DATA_FILE}"),
                device="cuda",
                models_dir=f"{OUTPUT_DIR}/models",
                tensorboard_log_dir=f"{OUTPUT_DIR}/runs/rosteals",
            )
        finally:
            # Persist checkpoints + logs to the Volume even if training errors out.
            output_volume.commit()


@app.local_entrypoint()
def run():
    train.remote()
