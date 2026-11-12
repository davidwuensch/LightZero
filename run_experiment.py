"""Train AlphaZero, MuZero and UniZero on Connect Four with identical budgets, then plot and summarise.

Every run uses the same env-step budget and MCTS simulations, and is evaluated every --eval-every env
steps with --eval-episodes games vs the rule bot (agent moves first).

Training modes (--modes):
    bot  the agent trains against the rule bot (all three algorithms)
    sp   self-play, the agent plays both sides (AlphaZero and MuZero only; UniZero is skipped)

Usage (from this directory):
    source /venv/main/bin/activate
    python run_experiment.py                                       # bot mode, 1000 steps, seed 0
    python run_experiment.py --modes bot,sp --seeds 0,1,2          # compare training modes too
    python run_experiment.py --algos alphazero --modes bot,sp --steps 5000 --eval-every 250 --seeds 0,1,2
    python run_experiment.py --algos alphazero,muzero --slip-prob 0.2 --seeds 0,1,2   # stochastic Connect Four
    python run_experiment.py --replot experiments/<name>           # only redo plots + summary

Output: experiments/<name>/
    runs/<algo>_<mode>_seed<N>/   TensorBoard logs, checkpoints (ckpt/final.pth.tar), progress.jsonl, run_info.json
    logs/<algo>_<mode>_seed<N>.log
    return_vs_steps.png    eval return vs env steps
    return_vs_games.png    eval return vs training games played (fairest x-axis across modes)
    return_vs_time.png     eval return vs wall-clock minutes
    summary.csv            per-run results

Note: an "env step" is one move in sp mode, but agent move + bot reply in bot mode, so for
bot-vs-sp comparisons prefer return_vs_games.png.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALGO_LABELS = {"alphazero": "AlphaZero", "muzero": "MuZero", "unizero": "UniZero"}
MODE_LABELS = {"bot": "vs bot", "sp": "self-play"}
SUPPORTED_MODES = {"alphazero": {"bot", "sp"}, "muzero": {"bot", "sp"}, "unizero": {"bot"}}
COLORS = {"alphazero": "C0", "muzero": "C1", "unizero": "C2"}
LINESTYLES = {"bot": "-", "sp": "--"}
TAG = "evaluator_step/eval_episode_return_mean"


# ----------------------------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------------------------
def run_jobs(jobs, parallel: int, exp_dir: str) -> dict:
    """Run training commands with at most `parallel` at once. Returns {job_name: exit_code}."""
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONWARNINGS="ignore")
    pending, running, results = list(jobs), {}, {}
    t_start = time.time()
    while pending or running:
        while pending and len(running) < parallel:
            name, cmd = pending.pop(0)
            log = open(os.path.join(exp_dir, "logs", f"{name}.log"), "w")
            running[name] = (subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
            print(f"  [start] {name}", flush=True)
        for name, (proc, log, t0) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                results[name] = proc.returncode
                status = "done " if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"  [{status}] {name} in {(time.time() - t0) / 60:.1f} min   "
                      f"({len(results)}/{len(jobs)} finished, {(time.time() - t_start) / 60:.1f} min elapsed)", flush=True)
                if proc.returncode != 0:
                    print(f"           see {os.path.join(exp_dir, 'logs', name + '.log')}", flush=True)
                del running[name]
        time.sleep(2)
    return results


# ----------------------------------------------------------------------------------------------
# Loading results
# ----------------------------------------------------------------------------------------------
def discover_runs(exp_dir: str) -> dict:
    """{(algo, mode): [run_dir, ...]} from each run's run_info.json."""
    groups = defaultdict(list)
    for info_path in sorted(glob.glob(os.path.join(exp_dir, "runs", "*", "run_info.json"))):
        info = json.load(open(info_path))
        groups[(info["algo"], info.get("mode", "bot"))].append(os.path.dirname(info_path))
    return dict(groups)


def load_run(run_dir: str):
    """Return dict(steps, returns, games, minutes) for a run's evaluations, or None."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    steps, vals = [], []
    for f in glob.glob(os.path.join(run_dir, "log", "serial", "events.out.tfevents.*")):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if TAG in ea.Tags()["scalars"]:
            for e in ea.Scalars(TAG):
                steps.append(e.step)
                vals.append(e.value)
    if not steps:
        return None
    o = np.argsort(steps)
    steps, vals = np.array(steps, dtype=float)[o], np.array(vals)[o]

    # Map env steps -> games and wall-clock via the per-collection progress log.
    games = minutes = None
    prog_path = os.path.join(run_dir, "progress.jsonl")
    if os.path.exists(prog_path):
        prog = [json.loads(l) for l in open(prog_path) if l.strip()]
        if prog:
            ps = np.array([0] + [p["envstep"] for p in prog], dtype=float)
            pg = np.array([0] + [p["episodes"] for p in prog], dtype=float)
            pt = np.array([0] + [p["wall_sec"] for p in prog], dtype=float) / 60
            games, minutes = np.interp(steps, ps, pg), np.interp(steps, ps, pt)
    return dict(steps=steps, returns=vals, games=games, minutes=minutes)


# ----------------------------------------------------------------------------------------------
# Plots and summary
# ----------------------------------------------------------------------------------------------
def label(algo, mode, modes_present, n):
    name = ALGO_LABELS[algo] + (f" ({MODE_LABELS[mode]})" if len(modes_present) > 1 else "")
    return f"{name}, {n} seed{'s' if n > 1 else ''}"


def plot(exp_dir: str, groups: dict, smooth: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    sys.path.insert(0, HERE)
    from plot_curves import smooth as ema

    data = {k: [r for r in (load_run(d) for d in dirs) if r] for k, dirs in groups.items()}
    modes_present = {m for (_, m) in groups}
    plt.style.use("seaborn-v0_8-darkgrid")

    for x_key, fname, xlabel in (("steps", "return_vs_steps.png", "Env Steps"),
                                 ("games", "return_vs_games.png", "Training games played"),
                                 ("minutes", "return_vs_time.png", "Wall-clock time (min)")):
        fig, ax = plt.subplots(figsize=(7, 5))
        drawn = 0
        for (algo, mode), runs in sorted(data.items()):
            runs = [r for r in runs if r[x_key] is not None]
            if not runs:
                continue
            xs = [r[x_key] for r in runs]
            x_max = min(x[-1] for x in xs)
            grid = np.unique(np.concatenate([x[x <= x_max] for x in xs]))
            ys = np.stack([np.interp(grid, x, ema(r["returns"], smooth)) for x, r in zip(xs, runs)])
            mean, std = ys.mean(0), ys.std(0)
            ax.plot(grid, mean, color=COLORS[algo], linestyle=LINESTYLES[mode],
                    marker="o" if len(grid) <= 30 else None, markersize=3,
                    label=label(algo, mode, modes_present, len(runs)))
            if len(runs) > 1:
                ax.fill_between(grid, mean - std, mean + std, color=COLORS[algo], alpha=0.15, linewidth=0)
            drawn += 1
        if not drawn:
            plt.close(fig)
            print(f"[skip] {fname}: no runs with '{x_key}' data")
            continue
        ax.set_title("Connect4 (evaluated vs rule bot)", fontsize=15)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Return", fontsize=13)
        ax.set_ylim(-1.05, 1.05)
        if x_key == "steps":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e5 else f"{x:g}"))
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        out = os.path.join(exp_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def summarise(exp_dir: str, groups: dict):
    rows = []
    for (algo, mode), dirs in sorted(groups.items()):
        for run in dirs:
            r = load_run(run)
            info = json.load(open(os.path.join(run, "run_info.json")))
            row = dict(algo=algo, mode=mode, run=os.path.basename(run), evals=0,
                       wall_min=round(info.get("wall_time_sec", float("nan")) / 60, 1))
            if r is not None:
                v = r["returns"]
                row.update(evals=len(v), last_eval_step=int(r["steps"][-1]),
                           games=int(r["games"][-1]) if r["games"] is not None else "",
                           final_return=round(float(v[-1]), 3), best_return=round(float(v.max()), 3),
                           mean_last3=round(float(v[-3:].mean()), 3))
            rows.append(row)

    path = os.path.join(exp_dir, "summary.csv")
    fields = ["algo", "mode", "run", "evals", "last_eval_step", "games", "final_return", "best_return",
              "mean_last3", "wall_min"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'algo':<11}{'mode':<11}{'seeds':>5}{'final return':>17}{'mean last 3':>16}{'best':>7}"
          f"{'games':>8}{'train time':>12}")
    for (algo, mode) in sorted(groups):
        r = [x for x in rows if x["algo"] == algo and x["mode"] == mode and x["evals"]]
        if not r:
            print(f"{ALGO_LABELS[algo]:<11}{MODE_LABELS[mode]:<11}{0:>5}   (no evaluation data)")
            continue
        fin = np.array([x["final_return"] for x in r]); l3 = np.array([x["mean_last3"] for x in r])
        best = np.array([x["best_return"] for x in r]); wt = np.array([x["wall_min"] for x in r])
        games = [x["games"] for x in r if x["games"] != ""]
        g = f"{np.mean(games):.0f}" if games else "-"
        print(f"{ALGO_LABELS[algo]:<11}{MODE_LABELS[mode]:<11}{len(r):>5}{fin.mean():>10.2f} ± {fin.std():.2f}"
              f"{l3.mean():>9.2f} ± {l3.std():.2f}{best.mean():>7.2f}{g:>8}{np.nanmean(wt):>8.1f} min")
    print(f"\nSaved {path}")


# ----------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", default="alphazero,muzero,unizero")
    p.add_argument("--modes", default="bot", help="comma-separated: bot, sp")
    p.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--sims", type=int, default=50)
    p.add_argument("--parallel", type=int, default=3, help="runs at once (they share the GPU)")
    p.add_argument("--slip-prob", type=float, default=0.0, help="stochastic env, passed to train_connect4.py")
    p.add_argument("--slip-to", choices=("adjacent", "any"), default="adjacent")
    p.add_argument("--sim-slip-prob", type=float, default=None, help="AlphaZero simulator slip (default: = --slip-prob)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="passed to train_connect4.py")
    p.add_argument("--collector-envs", type=int, default=None, help="games per process, passed to train_connect4.py")
    p.add_argument("--no-cuda-graphs", action="store_true", help="passed to train_connect4.py")
    p.add_argument("--name", default=None, help="experiment folder name (default: timestamp)")
    p.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing for plots, 0..1")
    p.add_argument("--replot", default=None, help="existing experiment dir: only redo plots + summary")
    args = p.parse_args()

    if args.replot:
        exp_dir = args.replot
    else:
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        for a in algos:
            if a not in ALGO_LABELS:
                raise SystemExit(f"Unknown algo '{a}'. Choose from {list(ALGO_LABELS)}")
        for m in modes:
            if m not in MODE_LABELS:
                raise SystemExit(f"Unknown mode '{m}'. Choose from {list(MODE_LABELS)}")
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

        name = args.name or datetime.now().strftime("c4_%y%m%d_%H%M%S")
        exp_dir = os.path.join("experiments", name)
        if os.path.exists(os.path.join(exp_dir, "runs")):
            raise SystemExit(f"{exp_dir} already exists; pick another --name or use --replot {exp_dir}")
        os.makedirs(exp_dir, exist_ok=True)
        json.dump(dict(vars(args), algos=algos, modes=modes, seeds=seeds, started=datetime.now().isoformat()),
                  open(os.path.join(exp_dir, "experiment.json"), "w"), indent=2)

        jobs, skipped = [], []
        for seed in seeds:
            for mode in modes:
                for algo in algos:
                    if mode not in SUPPORTED_MODES[algo]:
                        skipped.append(f"{algo}/{mode}")
                        continue
                    run = f"{algo}_{mode}_seed{seed}"
                    jobs.append((run, [
                        sys.executable, "train_connect4.py", "--algo", algo, "--mode", mode, "--seed", str(seed),
                        "--steps", str(args.steps), "--sims", str(args.sims),
                        "--eval-every", str(args.eval_every), "--eval-episodes", str(args.eval_episodes),
                        "--root", os.path.join(exp_dir, "runs"), "--exp-name", run, "--device", args.device,
                        "--slip-prob", str(args.slip_prob), "--slip-to", args.slip_to,
                        *(["--sim-slip-prob", str(args.sim_slip_prob)] if args.sim_slip_prob is not None else []),
                        *(["--collector-envs", str(args.collector_envs)] if args.collector_envs else []),
                        *(["--no-cuda-graphs"] if args.no_cuda_graphs else []),
                    ]))
        if skipped:
            print(f"Skipping unsupported combinations: {', '.join(sorted(set(skipped)))}")
        print(f"Experiment {exp_dir}: {len(jobs)} runs, {args.steps} env steps each, "
              f"eval every {args.eval_every} steps ({args.eval_episodes} games), {args.parallel} in parallel", flush=True)
        results = run_jobs(jobs, args.parallel, exp_dir)
        failed = [n for n, rc in results.items() if rc != 0]
        if failed:
            print(f"\n{len(failed)} run(s) failed: {', '.join(failed)}")

    groups = discover_runs(exp_dir)
    if not groups:
        raise SystemExit(f"No finished runs (run_info.json) found in {exp_dir}/runs")
    plot(exp_dir, groups, args.smooth)
    summarise(exp_dir, groups)


if __name__ == "__main__":
    main()
