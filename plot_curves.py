"""Plot LightZero training curves (mean over seeds ± shaded band), README style.

Each --run is "Label=glob": all run directories matching the glob are treated as seeds of one
curve. Without --run, every run under data_az/ is grouped by its name, with trailing
"_seedN" and LightZero's "_YYMMDD_HHMMSS" suffix stripped.

Usage (from this directory):
    source .venv/bin/activate
    python plot_curves.py                                          # all runs in data_az/
    python plot_curves.py --run "AlphaZero=data_az/c4_sims50_seed*" --run "AZ 200 sims=data_az/c4_sims200_seed*"
    python plot_curves.py --tag learner_step/total_loss_avg_avg --ylabel Loss
    python plot_curves.py --list-tags                              # show all logged metrics
"""
import argparse
import glob
import os
import re
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(run_dir: str, tag: str):
    """Return (steps, values) for a tag, merged over all event files in the run."""
    steps, values = [], []
    for f in sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                steps.append(e.step)
                values.append(e.value)
    if not steps:
        return None
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(values)[order]


def smooth(y: np.ndarray, weight: float) -> np.ndarray:
    """TensorBoard-style exponential moving average."""
    if weight <= 0 or len(y) == 0:
        return y
    out, last = np.empty_like(y, dtype=float), y[0]
    for i, v in enumerate(y):
        last = last * weight + (1 - weight) * v
        out[i] = last
    return out


def default_groups(root: str):
    groups = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "events.out.tfevents.*"), recursive=True):
            name = re.sub(r"_\d{6}_\d{6}$", "", os.path.basename(d))
            name = re.sub(r"_seed\d+$", "", name)
            groups[name].append(d)
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", default=[], help='"Label=glob" of run dirs (repeatable)')
    p.add_argument("--root", default="data_az", help="where to look when --run is not given")
    p.add_argument("--tag", default="evaluator_step/eval_episode_return_mean")
    p.add_argument("--title", default="Connect4")
    p.add_argument("--ylabel", default="Return")
    p.add_argument("--xlabel", default="Env Steps")
    p.add_argument("--smooth", type=float, default=0.6, help="EMA weight 0..1 (TensorBoard's slider)")
    p.add_argument("--band", choices=["std", "minmax", "none"], default="std")
    p.add_argument("--out", default="plots/curves.png")
    p.add_argument("--list-tags", action="store_true")
    args = p.parse_args()

    if args.run:
        groups = {}
        for spec in args.run:
            label, _, pattern = spec.partition("=")
            groups[label] = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    else:
        groups = default_groups(args.root)
    if not any(groups.values()):
        raise SystemExit("No runs found. Train first, or check your --run globs.")

    if args.list_tags:
        first = next(d for ds in groups.values() for d in ds)
        for f in glob.glob(os.path.join(first, "**", "events.out.tfevents.*"), recursive=True):
            ea = EventAccumulator(f, size_guidance={"scalars": 1})
            ea.Reload()
            print("\n".join(sorted(ea.Tags()["scalars"])))
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, dirs) in enumerate(groups.items()):
        curves = [c for c in (load_scalar(d, args.tag) for d in dirs) if c is not None]
        if not curves:
            print(f"[skip] {label}: no '{args.tag}' data in {len(dirs)} run(s)")
            continue
        # Put all seeds on a common x grid, only where every seed has data.
        x_max = min(s[-1] for s, _ in curves)
        grid = np.unique(np.concatenate([s[s <= x_max] for s, _ in curves]))
        ys = np.stack([np.interp(grid, s, smooth(v, args.smooth)) for s, v in curves])
        mean = ys.mean(axis=0)
        color = colors[i % len(colors)]
        ax.plot(grid, mean, color=color, label=f"{label} ({len(curves)} seed{'s' if len(curves) > 1 else ''})")
        if args.band != "none" and len(curves) > 1:
            lo, hi = ((mean - ys.std(0), mean + ys.std(0)) if args.band == "std" else (ys.min(0), ys.max(0)))
            ax.fill_between(grid, lo, hi, color=color, alpha=0.25, linewidth=0)
        print(f"{label}: {len(curves)} run(s), {len(grid)} points, final mean {mean[-1]:.3f}")

    ax.set_title(args.title, fontsize=16)
    ax.set_xlabel(args.xlabel, fontsize=14)
    ax.set_ylabel(args.ylabel, fontsize=14)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    ax.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
