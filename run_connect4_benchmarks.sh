#!/bin/bash
# AlphaZero vs MuZero on Connect Four: deterministic and stochastic ("slippery columns") environment.
#
#   1. trains (one experiment after the other)
#        experiments/c4_az_fixed : AlphaZero, deterministic env, 3 seeds  (MuZero deterministic = experiments/c4_az_mz)
#        experiments/c4_slip20   : AlphaZero + MuZero, slip_prob 0.2, 3 seeds each
#   2. plots the learning curves (mean over seeds +- std) into experiments/c4_benchmark/
#   3. benchmarks the seed-0 agents vs random / rule bot / pure MCTS / each other (tables in experiments/c4_benchmark/)
#
# Usage (from the repo root):   nohup bash run_connect4_benchmarks.sh > experiments/c4_benchmark.log 2>&1 &
# Re-running skips experiments whose summary.csv already exists, so it can also just redo plots + benchmarks.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

STEPS=${STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_EPISODES=${EVAL_EPISODES:-20}
SEEDS=${SEEDS:-0,1,2}
SLIP=${SLIP:-0.2}
SIMS=${SIMS:-50}
GAMES=${GAMES:-40}
DET_MUZERO=${DET_MUZERO:-experiments/c4_az_mz}   # finished deterministic MuZero runs
SUFFIX=${SUFFIX:-}                              # appended to all experiment/output names (e.g. for tests)
SLIPTAG=$(python3 -c "print(round($SLIP * 100))")   # 0.2 -> 20
OUT=experiments/c4_benchmark$SUFFIX
mkdir -p "$OUT"

common=(--seeds "$SEEDS" --steps "$STEPS" --eval-every "$EVAL_EVERY" --eval-episodes "$EVAL_EPISODES" --sims "$SIMS")

run() {   # run_experiment unless it already finished
    local name=$1; shift
    if [ -f "experiments/$name/summary.csv" ]; then
        echo "[skip] experiments/$name already finished"
    else
        echo "[$(date +%T)] start experiments/$name"
        python run_experiment.py --name "$name" "${common[@]}" "$@" && echo "[$(date +%T)] done experiments/$name"
    fi
}

echo "===== 1. training ($(date))"
# One experiment after the other: each run needs ~1-2 GB of GPU memory, and 9 at once overflowed the 16 GB RTX 5080.
run "c4_az_fixed$SUFFIX" --algos alphazero --parallel 3
run "c4_slip$SLIPTAG$SUFFIX" --algos alphazero,muzero --slip-prob "$SLIP" --parallel 6
SLIPDIR="experiments/c4_slip$SLIPTAG$SUFFIX/runs"
DETDIR="experiments/c4_az_fixed$SUFFIX/runs"

echo "===== 2. plots ($(date))"
plot() { python plot_curves.py --smooth 0.6 "$@"; }
plot --title "Connect4 (deterministic)" --out "$OUT/deterministic.png" \
     --run "AlphaZero=$DETDIR/alphazero_bot_seed*" \
     --run "MuZero=$DET_MUZERO/runs/muzero_bot_seed*"
plot --title "Connect4 (stochastic, slip $SLIP)" --out "$OUT/stochastic_slip$SLIPTAG.png" \
     --run "AlphaZero=$SLIPDIR/alphazero_bot_seed*" \
     --run "MuZero=$SLIPDIR/muzero_bot_seed*"
plot --title "Connect4: deterministic vs stochastic (slip $SLIP)" --out "$OUT/all.png" \
     --run "AlphaZero=$DETDIR/alphazero_bot_seed*" \
     --run "MuZero=$DET_MUZERO/runs/muzero_bot_seed*" \
     --run "AlphaZero slip $SLIP=$SLIPDIR/alphazero_bot_seed*" \
     --run "MuZero slip $SLIP=$SLIPDIR/muzero_bot_seed*"

echo "===== 3. benchmarks ($(date)): $GAMES games per opponent, $SIMS sims/move, seed-0 agents"
AZ_DET=$DETDIR/alphazero_bot_seed0/ckpt/final.pth.tar
MZ_DET=$DET_MUZERO/runs/muzero_bot_seed0/ckpt/final.pth.tar
AZ_SLIP=$SLIPDIR/alphazero_bot_seed0/ckpt/final.pth.tar
MZ_SLIP=$SLIPDIR/muzero_bot_seed0/ckpt/final.pth.tar
bench() {   # name ckpt opponent-ckpt [extra args]
    local name=$1 ckpt=$2 opp=$3; shift 3
    python benchmark_connect4.py --ckpt "$ckpt" --games "$GAMES" --sims "$SIMS" \
        --opponents "random,rule,mcts,ckpt:$opp" "$@" > "$OUT/benchmark_$name.txt" 2>&1
    echo "--- $name"; grep -A8 "^Agent:" "$OUT/benchmark_$name.txt"
}
bench alphazero_deterministic "$AZ_DET" "$MZ_DET" --slip-prob 0 &
bench muzero_deterministic "$MZ_DET" "$AZ_DET" --slip-prob 0 &
bench alphazero_slip "$AZ_SLIP" "$MZ_SLIP" &
bench muzero_slip "$MZ_SLIP" "$AZ_SLIP" &
wait
echo "===== finished ($(date)). Plots and benchmark tables: $OUT/"
touch "$OUT/.done"
