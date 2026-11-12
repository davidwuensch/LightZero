"""Benchmark a trained Connect Four AlphaZero agent against baseline opponents.

The agent plays N games against each opponent, half as first player and half as second.
The first few plies are random (--random-plies) so a deterministic agent doesn't replay
the same game over and over.

Usage (from this directory):
    source .venv/bin/activate
    python benchmark_connect4.py                                   # latest ckpt vs random, rule, mcts
    python benchmark_connect4.py --games 100 --opponents rule
    python benchmark_connect4.py --opponents mcts --mcts-sims 1000 # stronger pure-MCTS baseline
    python benchmark_connect4.py --opponents ckpt:data_az/connect4_alphazero/ckpt/iteration_250.pth.tar

Opponents:
    random        uniform random legal moves
    rule          LightZero's rule bot (win / block / avoid traps / build sequences)
    mcts          pure MCTS with random rollouts, no neural network (--mcts-sims)
    ckpt:<path>   another AlphaZero checkpoint (e.g. an earlier one), to measure progress
"""
import argparse
import contextlib
import copy
import io
import math
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from ding.config import compile_config
from ding.policy import create_policy
from easydict import EasyDict

from zoo.board_games.connect4.config.connect4_alphazero_sp_mode_config import main_config, create_config
from zoo.board_games.connect4.envs.connect4_env import Connect4Env
from zoo.board_games.connect4.envs.rule_bot import Connect4RuleBot
from zoo.board_games.mcts_bot import MCTSBot
from play_connect4 import latest_checkpoint


def load_agent(ckpt: str, sims: int):
    """Return (policy_eval_mode, env_cfg) for a checkpoint."""
    cfg = copy.deepcopy(main_config)
    cfg.policy.cuda = False
    cfg.policy.device = "cpu"
    cfg.policy.mcts_ctree = True
    cfg.env.alphazero_mcts_ctree = True
    cfg.env.battle_mode = "self_play_mode"
    cfg.policy.mcts.num_simulations = sims
    cfg.policy.collector_env_num = cfg.policy.evaluator_env_num = 1
    cfg.env.collector_env_num = cfg.env.evaluator_env_num = 1
    cfg.exp_name = "data_az/benchmark"
    cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=copy.deepcopy(create_config), save_cfg=False)
    cfg.policy.full_cfg = cfg
    cfg.policy.create_cfg = create_config
    policy = create_policy(cfg.policy, enable_field=["learn", "eval"])
    policy.learn_mode.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return policy.eval_mode, cfg.env


def make_player(spec: str, env_cfg, args):
    """Return a function (env, obs, ply) -> action for the given opponent spec."""
    if spec == "random":
        return lambda env, obs, ply: int(np.random.choice(env.legal_actions))
    if spec == "rule":
        def rule(env, obs, ply):
            return int(Connect4RuleBot(env, env.current_player).get_rule_bot_action(env.board, env.current_player))
        return rule
    if spec == "mcts":
        sim_cfg = EasyDict(copy.deepcopy(dict(env_cfg)))
        sim_cfg.alphazero_mcts_ctree = False
        sim_cfg.bot_action_type = None
        bot = MCTSBot(Connect4Env(sim_cfg), "mcts", args.mcts_sims)
        bot.plot_simulation_graph = lambda *a, **k: None   # upstream leaves debug graphviz plotting on

        def mcts(env, obs, ply):
            with contextlib.redirect_stdout(io.StringIO()):  # ...and a debug print per move
                action, *_ = bot.get_actions(np.array(env.board), ply, player_index=env.current_player_index,
                                             num_simulation=args.mcts_sims)  # returns (action, node, visits)
                return int(action)
        return mcts
    if spec.startswith("ckpt:"):
        other, _ = load_agent(spec[5:], args.sims)
        return lambda env, obs, ply: int(other.forward({0: obs})[0]["action"])
    raise SystemExit(f"Unknown opponent '{spec}'")


def play_game(env, agent_fn, opp_fn, agent_player: int, random_plies: int) -> int:
    """Return +1 agent win, 0 draw, -1 agent loss."""
    obs = env.reset()
    ply = 0
    while True:
        if ply < random_plies:
            action = int(np.random.choice(env.legal_actions))
        elif env.current_player == agent_player:
            action = agent_fn(env, obs, ply)
        else:
            action = opp_fn(env, obs, ply)
        ts = env.step(action)
        obs = ts.obs
        ply += 1
        if ts.done:
            _, winner = env.get_done_winner()
            return 0 if winner == -1 else (1 if winner == agent_player else -1)


def wilson(score: float, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion (draws counted as half a win)."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = (score + z * z / (2 * n)) / denom
    half = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def elo_diff(score: float) -> str:
    if score <= 0:
        return "-inf"
    if score >= 1:
        return "+inf"
    return f"{-400 * math.log10(1 / score - 1):+.0f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="agent checkpoint (default: latest)")
    p.add_argument("--opponents", default="random,rule,mcts", help="comma-separated list, see docstring")
    p.add_argument("--games", type=int, default=40, help="games per opponent (split evenly between sides)")
    p.add_argument("--sims", type=int, default=100, help="agent MCTS simulations per move")
    p.add_argument("--mcts-sims", type=int, default=200, help="simulations for the pure-MCTS opponent")
    p.add_argument("--random-plies", type=int, default=2, help="random opening moves for game diversity")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    ckpt = args.ckpt or latest_checkpoint()
    agent, env_cfg = load_agent(ckpt, args.sims)
    agent_fn = lambda env, obs, ply: int(agent.forward({0: obs})[0]["action"])
    env = Connect4Env(env_cfg)

    print(f"Agent: {ckpt}  ({args.sims} sims/move, {args.random_plies} random opening plies)\n")
    header = f"{'opponent':<22}{'W':>5}{'D':>5}{'L':>5}   {'score':>6}  {'95% CI':>13}  {'Elo Δ':>6}  {'as 1st':>7}  {'as 2nd':>7}  {'time':>6}"
    print(header)
    print("-" * len(header))

    for spec in [s.strip() for s in args.opponents.split(",") if s.strip()]:
        opp_fn = make_player(spec, env_cfg, args)
        results = {1: [], 2: []}
        t0 = time.time()
        for g in range(args.games):
            agent_player = 1 if g % 2 == 0 else 2
            results[agent_player].append(play_game(env, agent_fn, opp_fn, agent_player, args.random_plies))
        allr = results[1] + results[2]
        w, d, l = allr.count(1), allr.count(0), allr.count(-1)
        score = (w + 0.5 * d) / len(allr)
        lo, hi = wilson(score, len(allr))
        side = lambda r: f"{(r.count(1) + 0.5 * r.count(0)) / len(r):.0%}" if r else "-"
        name = spec if len(spec) <= 21 else "ckpt:…" + spec[-15:]
        print(f"{name:<22}{w:>5}{d:>5}{l:>5}   {score:>6.1%}  [{lo:>4.0%}, {hi:>4.0%}]  {elo_diff(score):>6}"
              f"  {side(results[1]):>7}  {side(results[2]):>7}  {time.time() - t0:>5.0f}s")

    print("\nscore = (wins + ½·draws) / games.  Elo Δ = agent strength relative to that opponent.")


if __name__ == "__main__":
    main()
