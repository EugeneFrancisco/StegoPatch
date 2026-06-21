"""Plot smoothed training curves from TensorBoard event files.

A single training run is often spread across several event files (one per
checkpoint / restart). Because those files describe the same run, their curves
line up and can be overlaid to give one continuous-looking plot per metric.
"""

import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth(values, weight):
    """Exponential moving average with bias correction (TensorBoard's smoothing).

    `weight` is in [0, 1): 0 is no smoothing, closer to 1 is smoother. This
    matches the curve produced by TensorBoard's smoothing slider.
    """
    last, debias, smoothed = 0.0, 0.0, []
    for v in values:
        last = last * weight + (1 - weight) * v
        debias = debias * weight + (1 - weight)  # corrects the cold-start bias
        smoothed.append(last / debias)
    return smoothed


def _read_scalar(event_file, metric):
    """Return (steps, values) for `metric` in `event_file`, or None if absent."""
    ea = EventAccumulator(event_file)
    ea.Reload()
    if metric not in ea.Tags()["scalars"]:
        return None
    events = ea.Scalars(metric)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def plot_metrics(event_files, metrics, save_dir, weight=0.6): # pylint: disable=redefined-outer-name
    """Plot each metric across all event files, one figure per metric.

    Args:
        event_files: list of (path, name) tuples. `path` is the TensorBoard
            event file to read and `name` is the label used for it in the plots.
        metrics: scalar tags to plot (e.g. "delta_l2", "loss/recovery").
        save_dir: directory the plots are written to (created if needed).
        weight: smoothing strength in [0, 1); shared across all metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    for metric in metrics:
        plt.figure()
        plotted = False

        for event_file, name in event_files:
            result = _read_scalar(event_file, metric)
            if result is None:
                print(f"warning: '{metric}' not found in {name}, skipping")
                continue
            steps, values = result
            # raw curve faint in the background, smoothed curve on top
            (line,) = plt.plot(steps, values, alpha=0.2, linewidth=1)
            plt.plot(
                steps,
                smooth(values, weight),
                color=line.get_color(),
                linewidth=2,
                label=name,
            )
            plotted = True

        if not plotted:
            print(f"warning: no data for '{metric}', skipping plot")
            plt.close()
            continue

        plt.title(metric)
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = metric.replace("/", "_") + ".png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved {filename}")


if __name__ == "__main__":
    runs = "results/experiment_1/runs"
    event_files = [
        (os.path.join(
                    runs,
                    "rosteals_2026-06-18_07-07-57",
                    "events.out.tfevents.1781766477.modal.2.0"
                    ),
        "Warmup 1"),
        (os.path.join(
                    runs,
                    "rosteals_2026-06-18_15-39-59",
                    "events.out.tfevents.1781797199.modal.2.0"
                    ),
        "Warmup 2"),
        (os.path.join(
                    runs,
                    "rosteals_2026-06-18_16-39-36",
                    "events.out.tfevents.1781800776.modal.2.0"
                    ),
        "Exposure 1"),
        (os.path.join(
                    runs,
                    "rosteals_2026-06-18_19-26-58",
                    "events.out.tfevents.1781810818.modal.2.0"
                    ),
        "Exposure 2"),
        (os.path.join(
                    runs,
                    "rosteals_2026-06-21_02-20-51",
                    "events.out.tfevents.1782008451.modal.2.0"
                    ),
        "Noising"),
    ]
    metrics = ["delta_l2", "loss/recovery", "loss/quality", "bit_accuracy"]
    plot_metrics(event_files, metrics, "results/experiment_1/plots")
