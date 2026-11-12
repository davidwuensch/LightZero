"""Play Connect Four against your trained AlphaZero agent in the terminal.

Usage (from this directory):
    source .venv/bin/activate
    python play_connect4.py                       # latest checkpoint, you move first
    python play_connect4.py --second              # the agent moves first
    python play_connect4.py --sims 200            # stronger (slower) agent search
    python play_connect4.py --ckpt data_az/connect4_alphazero/ckpt/iteration_250.pth.tar
"""
import argparse
import glob
import os
import re
import warnings

warnings.filterwarnings("ignore")

import torch
from ding.config import compile_config
from ding.policy import create_policy

from zoo.board_games.connect4.config.connect4_alphazero_sp_mode_config import main_config, create_config
from zoo.board_games.connect4.envs.connect4_env import Connect4Env

SYMBOLS = {0: "·", 1: "\033[91m●\033[0m", 2: "\033[93m●\033[0m"}  # empty, red, yellow


def latest_checkpoint() -> str:
    ckpts = glob.glob("data_az/*/ckpt/*.pth.tar")
    if not ckpts:
        raise SystemExit("No checkpoint found under data_az/. Train first with: python train_connect4.py")
    best = [c for c in ckpts if c.endswith("ckpt_best.pth.tar")]
    if best:
        return max(best, key=os.path.getmtime)
    return max(ckpts, key=lambda c: int(re.search(r"iteration_(\d+)", c).group(1)) if "iteration_" in c else -1)


def show(board) -> None:
    rows = [list(board[r * 7:(r + 1) * 7]) for r in range(6)]
    print()
    for row in rows:
        print("  " + " ".join(SYMBOLS[int(v)] for v in row))
    print("  " + " ".join(str(c) for c in range(1, 8)))
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: latest)")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per agent move")
    parser.add_argument("--second", action="store_true", help="let the agent move first")
    args = parser.parse_args()

    ckpt = args.ckpt or latest_checkpoint()

    main_config.policy.cuda = False
    main_config.policy.mcts_ctree = True
    main_config.env.alphazero_mcts_ctree = True
    main_config.env.battle_mode = "self_play_mode"   # we drive both sides ourselves
    main_config.policy.mcts.num_simulations = args.sims
    main_config.policy.collector_env_num = 1
    main_config.policy.evaluator_env_num = 1
    main_config.env.collector_env_num = 1
    main_config.env.evaluator_env_num = 1
    main_config.exp_name = "data_az/play"
    main_config.policy.device = "cpu"

    cfg = compile_config(main_config, seed=0, env=None, auto=True, create_cfg=create_config, save_cfg=False)
    cfg.policy.full_cfg = cfg
    cfg.policy.create_cfg = create_config
    policy = create_policy(cfg.policy, enable_field=["learn", "eval"])
    policy.learn_mode.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agent = policy.eval_mode

    env = Connect4Env(cfg.env)
    obs = env.reset()

    human_player = 2 if args.second else 1
    print(f"Loaded {ckpt}")
    print(f"You are {SYMBOLS[human_player]}  (player {human_player}). Enter a column 1-7, or q to quit.")

    done = False
    while not done:
        show(env.board)
        if env.current_player == human_player:
            while True:
                s = input("Your move: ").strip().lower()
                if s in ("q", "quit", "exit"):
                    return
                if s.isdigit() and int(s) - 1 in env.legal_actions:
                    action = int(s) - 1
                    break
                print(f"Pick a non-full column from {[a + 1 for a in env.legal_actions]}.")
        else:
            action = int(agent.forward({0: obs})[0]["action"])
            print(f"Agent plays column {action + 1}")
        timestep = env.step(action)
        obs, done = timestep.obs, timestep.done

    show(env.board)
    _, winner = env.get_done_winner()
    if winner == -1:
        print("Draw!")
    elif winner == human_player:
        print("You win! 🎉")
    else:
        print("The agent wins.")


if __name__ == "__main__":
    main()
