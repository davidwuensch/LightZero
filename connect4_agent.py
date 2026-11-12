"""Shared helpers for the Connect Four scripts: checkpoint discovery, device selection and an
algorithm-agnostic agent loader (AlphaZero and MuZero checkpoints).

    from connect4_agent import load_agent, latest_checkpoint, resolve_device
    agent = load_agent(latest_checkpoint(), sims=100, device="auto")
    action = agent.act(env, obs)      # env = agent.make_env(), obs = env.reset()
"""
import copy
import glob
import json
import os

import numpy as np
import torch

# Where train_connect4.py (data_c4/<run>) and run_experiment.py (experiments/<name>/runs/<run>) put runs.
DEFAULT_ROOT = "data_c4"
RUN_GLOBS = (os.path.join(DEFAULT_ROOT, "*"), os.path.join("experiments", "*", "runs", "*"))


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' if a GPU is available, else 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch sees no GPU.")
    return device


def latest_checkpoint() -> str:
    """Newest final/best checkpoint of any run; falls back to the newest iteration checkpoint."""
    ckpts = [c for g in RUN_GLOBS for c in glob.glob(os.path.join(g, "ckpt", "*.pth.tar"))]
    if not ckpts:
        raise SystemExit(f"No checkpoint found under {DEFAULT_ROOT}/ or experiments/. "
                         "Train first with: python train_connect4.py")
    preferred = [c for c in ckpts if os.path.basename(c) in ("final.pth.tar", "ckpt_best.pth.tar")]
    return max(preferred or ckpts, key=os.path.getmtime)


def run_info(ckpt: str) -> dict:
    """The run_info.json written by train_connect4.py next to the checkpoint's ckpt/ folder ({} if missing)."""
    info_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))), "run_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            return json.load(f)
    return {}


def checkpoint_info(ckpt: str, state_dict: dict) -> tuple:
    """(algo, mode) of a checkpoint: from the run's run_info.json if present, else guessed from its keys."""
    info = run_info(ckpt)
    if info:
        return info["algo"], info.get("mode", "bot")
    keys = state_dict.get("model", {}).keys()
    if any(k.startswith("world_model") or k.startswith("tokenizer") for k in keys):
        algo = "unizero"
    elif "target_model" in state_dict:
        algo = "muzero"
    else:
        algo = "alphazero"
    return algo, "bot"  # mode is only needed for MuZero; bot is the training default


class Agent:
    """A loaded checkpoint that picks moves for whichever player is to move in a self-play-mode env."""

    def __init__(self, ckpt: str, sims: int = 100, device: str = "auto", algo: str = None, mode: str = None,
                 slip_prob: float = None, slip_to: str = None, sim_slip_prob: float = None):
        """slip_prob / slip_to: stochastic dynamics of the env made by ``make_env()``; sim_slip_prob: AlphaZero's
        MCTS simulator. Each defaults to the value the checkpoint was trained with (run_info.json), else 0 / adjacent."""
        from ding.config import compile_config
        from ding.policy import create_policy

        self.ckpt = ckpt
        self.device = resolve_device(device)
        state_dict = torch.load(ckpt, map_location=self.device, weights_only=False)
        guessed_algo, guessed_mode = checkpoint_info(ckpt, state_dict)
        self.algo, self.mode = algo or guessed_algo, mode or guessed_mode
        trained = run_info(ckpt)
        self.trained_slip_prob = float(trained.get("slip_prob") or 0.0)
        self.slip_prob = self.trained_slip_prob if slip_prob is None else slip_prob
        self.slip_to = slip_to or trained.get("slip_to") or "adjacent"
        if sim_slip_prob is None:
            sim_slip_prob = trained.get("sim_slip_prob")
            if sim_slip_prob is None:   # trained with simulator = env: keep that relation for the new env
                sim_slip_prob = self.slip_prob
        self.sim_slip_prob = float(sim_slip_prob)

        if self.algo == "alphazero":
            from zoo.board_games.connect4.config.connect4_alphazero_sp_mode_config import main_config, create_config
        elif self.algo == "muzero":
            from zoo.board_games.connect4.config.connect4_muzero_sp_mode_config import main_config, create_config
        else:
            raise SystemExit(f"{ckpt}: playing/benchmarking '{self.algo}' checkpoints is not supported "
                             "(only alphazero and muzero).")

        cfg, create = copy.deepcopy(main_config), copy.deepcopy(create_config)
        cfg.policy.cuda = self.device == "cuda"
        cfg.policy.device = self.device
        cfg.policy.model.use_cuda_graphs = self.device == "cuda"   # CUDA-graph replay of MCTS inference
        cfg.policy.mcts_ctree = True
        cfg.env.battle_mode = "self_play_mode"      # we drive both sides ourselves
        cfg.env.slip_prob = self.slip_prob
        cfg.env.slip_to = self.slip_to
        cfg.env.simulation_slip_prob = self.sim_slip_prob
        cfg.env.collector_env_num = cfg.env.evaluator_env_num = 1
        cfg.policy.collector_env_num = cfg.policy.evaluator_env_num = 1
        if self.algo == "alphazero":
            cfg.env.alphazero_mcts_ctree = True
            cfg.policy.mcts.num_simulations = sims
            cfg.policy.mcts.eval_num_simulations = sims   # upstream default would be 4 * sims in eval mode
        else:
            cfg.policy.num_simulations = sims
        cfg.exp_name = os.path.join(DEFAULT_ROOT, ".agent")

        cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create, save_cfg=False)
        cfg.policy.full_cfg = cfg
        cfg.policy.create_cfg = create
        self.policy = create_policy(cfg.policy, enable_field=["learn", "eval"])
        self.policy.learn_mode.load_state_dict(state_dict)
        self.policy._model.eval()   # inference: BatchNorm in eval mode, MuZero returns scalar values
        self.env_cfg = cfg.env

    def make_env(self):
        from zoo.board_games.connect4.envs.connect4_env import Connect4Env
        return Connect4Env(copy.deepcopy(self.env_cfg))

    def act(self, env, obs) -> int:
        """Action for the player to move. `obs` is the dict returned by a self-play-mode Connect4Env."""
        if self.algo == "alphazero":
            return int(self.policy.eval_mode.forward({0: obs})[0]["action"])
        x = torch.as_tensor(np.asarray(obs["observation"]), dtype=torch.float32, device=self.device)[None]
        # Bot-mode MuZero was trained single-agent (to_play=-1: no value sign flips in its search);
        # self-play MuZero alternates players in its search.
        to_play = [-1] if self.mode == "bot" else [int(obs["to_play"])]
        out = self.policy.eval_mode.forward(x, [np.asarray(obs["action_mask"])], to_play,
                                            ready_env_id=np.array([0]), timestep=[0])
        return int(out[0]["action"])

    def describe(self) -> str:
        s = f"{self.algo} ({self.mode} mode) on {self.device}"
        if self.slip_prob > 0 or self.trained_slip_prob > 0:
            s += f", env slip {self.slip_prob:g} ({self.slip_to}), trained with {self.trained_slip_prob:g}"
            if self.algo == "alphazero":
                s += f", simulator slip {self.sim_slip_prob:g}"
        return s


def load_agent(ckpt: str, sims: int = 100, device: str = "auto", algo: str = None, mode: str = None,
               slip_prob: float = None, slip_to: str = None, sim_slip_prob: float = None) -> Agent:
    return Agent(ckpt, sims, device, algo, mode, slip_prob, slip_to, sim_slip_prob)
