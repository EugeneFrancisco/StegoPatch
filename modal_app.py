"""
Modal entrypoint: trains the StegoPatch watermarker on a cloud GPU and exposes a
live TensorBoard dashboard over a Modal tunnel while it runs.

StegoPatch watermarks 384x384 images in 96x96 patches, so it trains on the
384-resolution precomputed array (the 256 array used for plain RoSteALS won't
work: encode_batch asserts the image side is divisible by the patch size).

One-time setup (upload the precomputed 384 training array to a Volume):

    modal volume put rosteals-data \\
        data/train2017_numpy_384.npy train2017_numpy_384.npy

Run training (prints a public TensorBoard URL near the top of the logs):

    modal run modal_app.py::run

Fetch the trained weights afterwards (timestamp is printed at the end of the run):

    modal volume get rosteals-output models/stegopatch_<timestamp>/checkpoint5.pt ./checkpoint5.pt

Resume training from a saved checkpoint (upload it first, then resume):

    modal volume put rosteals-output ./my_checkpoint.pt restart/checkpoint.pt
    modal run modal_app.py::resume                  # uses restart/checkpoint.pt, checkpoint 1
    modal run modal_app.py::resume --checkpoint 2

Validate a checkpoint on the test set (results land in the rosteals-output volume;
upload the 384 test .npy to rosteals-data under the same file name first):

    modal run modal_app.py::evaluate
    modal run modal_app.py::evaluate --checkpoint-path /output/models/stegopatch_<ts>/checkpoint5.pt
    modal volume get rosteals-output test_results.txt ./test_results.txt
"""
import subprocess
from pathlib import Path

import modal

APP_NAME = "stegopatch-watermark"

# Holds the precomputed (N, 3, 384, 384) uint8 .npy training array. The Volume
# name is kept as "rosteals-data" so existing uploads/credentials still resolve.
data_volume = modal.Volume.from_name("rosteals-data", create_if_missing=True)
# Holds training outputs: model checkpoints and TensorBoard run logs.
output_volume = modal.Volume.from_name("rosteals-output", create_if_missing=True)

DATA_DIR = "/data"
OUTPUT_DIR = "/output"
DATA_FILE = "train2017_numpy_384.npy"
TEST_FILE = "test2017_numpy_384.npy"

# Drop the checkpoint you want to resume from here, in the rosteals-output volume:
#
#     modal volume put rosteals-output ./my_checkpoint.pt restart/checkpoint.pt
#
# It then lives at this in-container path, which `modal run modal_app.py::restart`
# loads from by default.
RESTART_CHECKPOINT = f"{OUTPUT_DIR}/restart/checkpoint.pt"

# Checkpoint that `modal run modal_app.py::evaluate` validates by default, and the
# file the per-noise-type validation results get written to (both in rosteals-output).
TEST_CHECKPOINT = f"{OUTPUT_DIR}/restart/checkpoint.pt"
TEST_RESULTS = f"{OUTPUT_DIR}/test_results.txt"


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
        "scipy",
        "pillow",
        "tqdm",
        "tensorboard",
        "accelerate",
        "matplotlib",
        # ImageNet-C corruptions are reimplemented in src/noisers/imagenet_corruptions.py
        # on top of these, so no ImageMagick / imagenet_c system deps are needed.
        "scikit-image",
        "opencv-python-headless",
    )
    .run_function(_download_pretrained_weights)
    # Ship the local source last so editing it doesn't bust the build cache above.
    .add_local_python_source("src", "main")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="A100",
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
                tensorboard_log_dir=f"{OUTPUT_DIR}/runs/stegopatch",
            )
        finally:
            # Persist checkpoints + logs to the Volume even if training errors out.
            output_volume.commit()


@app.function(
    gpu="A100",
    volumes={DATA_DIR: data_volume, OUTPUT_DIR: output_volume},
    timeout=24 * 60 * 60,
)
def restart(save_path: str = RESTART_CHECKPOINT, checkpoint: int = 1):
    import main as train_main

    subprocess.Popen(
        ["tensorboard", "--logdir", f"{OUTPUT_DIR}/runs",
         "--host", "0.0.0.0", "--port", "6006"]
    )

    with modal.forward(6006) as tunnel:
        print(f"\n>>> TensorBoard live at: {tunnel.url}\n", flush=True)
        try:
            train_main.restart(
                save_path=save_path,
                checkpoint=checkpoint,
                data_path=Path(f"{DATA_DIR}/{DATA_FILE}"),
                device="cuda",
                models_dir=f"{OUTPUT_DIR}/models",
                tensorboard_log_dir=f"{OUTPUT_DIR}/runs/stegopatch",
            )
        finally:
            output_volume.commit()


@app.function(
    gpu="A100",
    volumes={DATA_DIR: data_volume, OUTPUT_DIR: output_volume},
    timeout=60 * 60,
)
def test(checkpoint_path: str = TEST_CHECKPOINT, results_path: str = TEST_RESULTS):
    import main as train_main

    # Build a StegoPatch pointed at the test .npy in the data volume, load the
    # checkpoint, and run the per-noise-type validation.
    stegopatch = train_main._build_stegopatch(
        data_path=Path(f"{DATA_DIR}/{TEST_FILE}"),
        device="cuda",
        models_dir=f"{OUTPUT_DIR}/models",
        tensorboard_log_dir=f"{OUTPUT_DIR}/runs/stegopatch",
        test_set_path=Path(f"{DATA_DIR}/{TEST_FILE}"),
    )
    stegopatch.load_model(checkpoint_path)
    results = stegopatch.validate()
    print(results)

    try:
        with open(results_path, "w", encoding="utf-8") as f:
            for name, value in results.items():
                f.write(f"{name}: {value}\n")
        print(f"Wrote validation results to {results_path}")
    finally:
        output_volume.commit()


@app.local_entrypoint()
def run():
    # .spawn() (not .remote()) so the run is fire-and-forget: it returns immediately
    # and is not tied to — or cancelled with — the local client (works with --detach).
    train.spawn()


@app.local_entrypoint()
def resume(save_path: str = RESTART_CHECKPOINT, checkpoint: int = 1):
    """Resume training from a checkpoint .pt file living in the rosteals-output volume.

    Usage (after uploading your checkpoint, see RESTART_CHECKPOINT above):

        modal run modal_app.py::resume
        modal run modal_app.py::resume --checkpoint 2
        modal run modal_app.py::resume --save-path /output/models/rosteals_.../checkpoint2.pt

    To run detached (survives client disconnect), add --detach:

        modal run --detach modal_app.py::resume

    .spawn() (not .remote()) is used so the run is fire-and-forget: it returns
    immediately and the call is not tied to — or cancelled with — the local client.
    """
    restart.spawn(save_path=save_path, checkpoint=checkpoint)


@app.local_entrypoint()
def evaluate(checkpoint_path: str = TEST_CHECKPOINT, results_path: str = TEST_RESULTS):
    """Run per-noise-type validation on a checkpoint living in the rosteals-output volume.

    Results are written to a text file in the rosteals-output volume.

    Usage:

        modal run modal_app.py::evaluate
        modal run modal_app.py::evaluate --checkpoint-path /output/models/rosteals_.../checkpoint5.pt
    """
    test.spawn(checkpoint_path=checkpoint_path, results_path=results_path)
