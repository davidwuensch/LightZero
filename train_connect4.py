"""Train AlphaZero, MuZero or UniZero on Connect Four (GPU if available, else CPU).

Usage (from this directory):
    source /venv/main/bin/activate
    python train_connect4.py --algo alphazero --steps 1000 --eval-every 100
    python train_connect4.py --algo muzero    --steps 1000 --eval-every 100 --seed 1
    python train_connect4.py --algo unizero   --steps 1000 --eval-every 100
    python train_connect4.py --algo alphazero --mode sp     # self-play training (alphazero/muzero only)
    python train_connect4.py --algo muzero --device cpu     # force CPU (default: --device auto = GPU if present)
    python train_connect4.py --algo alphazero --slip-prob 0.2   # stochastic Connect Four (pieces slip sideways)

Modes:
    bot  (default) the agent trains by playing against LightZero's rule bot ("play_with_bot_mode").
                   Supported by all three algorithms, so this is the mode for comparisons.
    sp   self-play: the agent plays both sides. AlphaZero and MuZero only.

Evaluation (all algorithms, all modes): the agent plays --eval-episodes games against the rule bot,
moving first; return = mean of +1 win / 0 draw / -1 loss. Logged to TensorBoard as
evaluator_step/eval_episode_return_mean. Evaluations run every --eval-every *env steps*
(at the first collection boundary after each multiple), so curves are comparable across algorithms.

Stochastic Connect Four: with --slip-prob p a dropped piece lands in a random non-full neighbouring column
(--slip-to any: any other column) with probability p, for every move of both players. AlphaZero's MCTS simulator
uses --sim-slip-prob (default: same as --slip-prob = true stochastic model; 0 = plans as if deterministic).
MuZero/UniZero learn their model from data, so --sim-slip-prob does not apply to them.

Output: <root>/<exp-name>/ with log/ (TensorBoard), ckpt/ (incl. final.pth.tar) and run_info.json.
"""
import argparse
import copy
import json
import os

# Several runs share one GPU: let the CUDA caching allocator grow/shrink segments instead of fragmenting
# (must be set before torch initialises CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import warnings

warnings.filterwarnings("ignore")

from easydict import EasyDict

from connect4_agent import DEFAULT_ROOT, resolve_device

ALGOS = ("alphazero", "muzero", "unizero")


# ----------------------------------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------------------------------
def alphazero_config(mode: str):
    if mode == "bot":
        from zoo.board_games.connect4.config.connect4_alphazero_bot_mode_config import main_config, create_config
    else:
        from zoo.board_games.connect4.config.connect4_alphazero_sp_mode_config import main_config, create_config
    from lzero.entry import train_alphazero
    main, create = copy.deepcopy(main_config), copy.deepcopy(create_config)
    main.policy.mcts_ctree = True            # compiled C++ MCTS (Python MCTS is broken upstream)
    main.env.alphazero_mcts_ctree = True
    return main, create, train_alphazero


def muzero_config(mode: str):
    if mode == "bot":
        from zoo.board_games.connect4.config.connect4_muzero_bot_mode_config import main_config, create_config
    else:
        from zoo.board_games.connect4.config.connect4_muzero_sp_mode_config import main_config, create_config
    from lzero.entry import train_muzero
    return copy.deepcopy(main_config), copy.deepcopy(create_config), train_muzero


def unizero_config(mode: str):
    """LightZero ships no board-game UniZero config; this one adapts the CartPole UniZero config
    (vector obs, small transformer) to Connect Four with the MuZero board-game settings."""
    if mode != "bot":
        raise SystemExit("UniZero is only set up for --mode bot (its two-player self-play path is untested upstream).")
    from lzero.entry import train_unizero
    collector_env_num, evaluator_env_num = 8, 5
    max_moves = 6 * 7 // 2  # agent moves per game in play_with_bot_mode
    main = EasyDict(dict(
        env=dict(
            battle_mode="play_with_bot_mode",
            bot_action_type="rule",
            channel_last=False,
            flatten_observation=True,           # (3, 6, 7) -> 126-dim vector for the MLP encoder
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False),
        ),
        policy=dict(
            learn=dict(learner=dict(hook=dict(save_ckpt_after_iter=1000000))),
            model=dict(
                observation_shape=126,
                action_space_size=7,
                self_supervised_learning_loss=True,
                discrete_action_encoding_type="one_hot",
                norm_type="BN",
                model_type="mlp",
                world_model_cfg=dict(
                    final_norm_option_in_obs_head="LayerNorm",
                    final_norm_option_in_encoder="LayerNorm",
                    predict_latent_loss_type="mse",
                    max_blocks=10,
                    max_tokens=2 * 10,
                    context_length=2 * 4,
                    context_length_for_recurrent=2 * 4,
                    device="cpu",               # overwritten from --device in main()
                    action_space_size=7,
                    num_layers=2,
                    num_heads=2,
                    embed_dim=64,
                    env_num=max(collector_env_num, evaluator_env_num),
                    collector_env_num=collector_env_num,
                    evaluator_env_num=evaluator_env_num,
                    obs_type="vector",
                    norm_type="BN",
                    rotary_emb=False,   # RoPE needs a 'timestep' obs key that board-game envs don't provide
                ),
            ),
            model_path=None,
            use_wandb=False,
            cuda=False,
            use_augmentation=False,
            env_type="board_games",
            action_type="varied_action_space",
            num_unroll_steps=5,
            td_steps=max_moves,
            discount_factor=1,
            game_segment_length=max_moves,
            # Same update budget as the AlphaZero/MuZero Connect4 configs (50 updates of 256 per collect)
            # instead of UniZero's default replay_ratio, so the comparison is fair per env step.
            update_per_collect=50,
            replay_ratio=0.25,  # only used if update_per_collect is None
            batch_size=256,
            optim_type="AdamW",
            piecewise_decay_lr_scheduler=False,
            learning_rate=1e-4,
            target_update_freq=100,
            grad_clip_value=5,
            num_simulations=50,
            reanalyze_ratio=0,
            n_episode=8,
            train_start_after_envsteps=0,
            eval_freq=int(1e3),
            replay_buffer_size=int(1e6),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
    ))
    create = EasyDict(dict(
        env=dict(type="connect4", import_names=["zoo.board_games.connect4.envs.connect4_env"]),
        env_manager=dict(type="subprocess"),
        policy=dict(type="unizero", import_names=["lzero.policy.unizero"]),
    ))
    return main, create, train_unizero


CONFIGS = {"alphazero": alphazero_config, "muzero": muzero_config, "unizero": unizero_config}


# ----------------------------------------------------------------------------------------------
# Evaluate every N env steps (LightZero's eval_freq counts training iterations, which differ per algorithm)
# ----------------------------------------------------------------------------------------------
def evaluate_every_env_steps(every: int) -> None:
    from lzero.worker.alphazero_evaluator import AlphaZeroEvaluator
    from lzero.worker.muzero_evaluator import MuZeroEvaluator  # also used by UniZero

    for cls in (AlphaZeroEvaluator, MuZeroEvaluator):
        original_eval = cls.eval

        def eval_(self, save_ckpt_fn=None, train_iter=-1, envstep=-1, *args, _orig=original_eval, **kwargs):
            last = getattr(self, "_last_eval_envstep", None)
            if last is not None and envstep // every <= last // every:
                return False, None          # no new multiple of `every` crossed since the last eval
            self._last_eval_envstep = envstep
            return _orig(self, save_ckpt_fn, train_iter, envstep, *args, **kwargs)

        cls.eval = eval_
        cls.should_eval = lambda self, train_iter: True


# ----------------------------------------------------------------------------------------------
# Record (env steps, games) after every collection, so curves can also be plotted per game played.
# An "env step" means different things per mode (sp: every move; bot: agent move + bot reply).
# ----------------------------------------------------------------------------------------------
def record_progress(path: str) -> None:
    from lzero.worker.alphazero_collector import AlphaZeroCollector
    from lzero.worker.muzero_collector import MuZeroCollector  # also used by UniZero
    t0 = time.time()

    for cls in (AlphaZeroCollector, MuZeroCollector):
        original_collect = cls.collect

        def collect_(self, *args, _orig=original_collect, **kwargs):
            out = _orig(self, *args, **kwargs)
            with open(path, "a") as f:
                f.write(json.dumps(dict(envstep=int(self._total_envstep_count),
                                        episodes=int(self._total_episode_count),
                                        wall_sec=round(time.time() - t0, 1))) + "\n")
            return out

        cls.collect = collect_


# ----------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, default="alphazero")
    p.add_argument("--mode", choices=("bot", "sp"), default="bot")
    p.add_argument("--steps", type=int, default=int(1e5), help="max environment steps (agent moves)")
    p.add_argument("--sims", type=int, default=50, help="MCTS simulations per move (training and evaluation)")
    p.add_argument("--eval-every", type=int, default=2000, help="evaluate every N env steps")
    p.add_argument("--eval-episodes", type=int, default=10, help="games vs the rule bot per evaluation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=DEFAULT_ROOT, help="parent folder for runs")
    p.add_argument("--exp-name", default=None, help="run folder name (default: <algo>_<mode>_seed<seed>)")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--slip-prob", type=float, default=0.0, help="stochastic env: probability a piece slips columns")
    p.add_argument("--slip-to", choices=("adjacent", "any"), default="adjacent", help="where a slipping piece lands")
    p.add_argument("--sim-slip-prob", type=float, default=None,
                   help="AlphaZero only: slip probability in its MCTS simulator (default: = --slip-prob)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="network training/inference device (auto = cuda if available)")
    p.add_argument("--no-cuda-graphs", action="store_true",
                   help="disable CUDA-graph replay of MCTS inference (alphazero/muzero; on by default on GPU)")
    p.add_argument("--collector-envs", type=int, default=None,
                   help="games collected in parallel per process (default 8). More = bigger GPU inference batches; "
                        "update_per_collect is scaled along so updates per collected game stay the same")
    args = p.parse_args()

    main_config, create_config, train_fn = CONFIGS[args.algo](args.mode)

    # Device: LightZero's train_* entries pick policy.device from policy.cuda; UniZero's world model
    # additionally needs its own device setting.
    device = resolve_device(args.device)
    main_config.policy.cuda = device == "cuda"
    main_config.policy.device = device
    if "world_model_cfg" in main_config.policy.get("model", {}):
        main_config.policy.model.world_model_cfg.device = device
    use_graphs = device == "cuda" and not args.no_cuda_graphs and args.algo in ("alphazero", "muzero")
    if args.algo in ("alphazero", "muzero"):
        main_config.policy.model.use_cuda_graphs = use_graphs
    if device == "cuda":
        import torch
        torch.backends.cudnn.benchmark = True   # fixed input shapes: let cuDNN pick the fastest conv kernels
    print(f"[train_connect4] {args.algo}/{args.mode} on {device}"
          + (" (CUDA-graph MCTS inference)" if use_graphs else ""), flush=True)

    # Parallel games per process (bigger MCTS inference batches), keeping updates per collected game constant.
    if args.collector_envs:
        if args.algo == "unizero":
            raise SystemExit("--collector-envs is not supported for unizero (its world model sizes its KV caches by env count).")
        old_n = main_config.env.collector_env_num
        scale = args.collector_envs / old_n
        main_config.env.collector_env_num = args.collector_envs
        main_config.policy.collector_env_num = args.collector_envs
        main_config.policy.n_episode = args.collector_envs
        main_config.policy.update_per_collect = max(1, round(main_config.policy.update_per_collect * scale))
        print(f"[train_connect4] {args.collector_envs} collector envs (was {old_n}), "
              f"update_per_collect={main_config.policy.update_per_collect}", flush=True)

    # Stochastic dynamics (all algorithms; simulation_slip_prob is only used by AlphaZero's simulator)
    main_config.env.slip_prob = args.slip_prob
    main_config.env.slip_to = args.slip_to
    main_config.env.simulation_slip_prob = args.sim_slip_prob
    if args.slip_prob > 0 or args.sim_slip_prob:
        sim = args.slip_prob if args.sim_slip_prob is None else args.sim_slip_prob
        print(f"[train_connect4] stochastic env: slip_prob={args.slip_prob} ({args.slip_to})"
              + (f", AlphaZero simulator slip_prob={sim}" if args.algo == "alphazero" else ""), flush=True)

    # Search budget
    if args.algo == "alphazero":
        main_config.policy.mcts.num_simulations = args.sims
        # LightZero's AlphaZero evaluates with min(800, 4 * sims) by default, MuZero/UniZero with sims:
        # use the same budget so the evaluation curves are comparable (and evaluation is 4x cheaper).
        main_config.policy.mcts.eval_num_simulations = args.sims
    else:
        main_config.policy.num_simulations = args.sims

    # Evaluation: same opponent and number of games for every algorithm; never stop early.
    n_eval = args.eval_episodes
    main_config.env.evaluator_env_num = n_eval
    main_config.env.n_evaluator_episode = n_eval
    main_config.policy.evaluator_env_num = n_eval
    if "world_model_cfg" in main_config.policy.get("model", {}):
        wm = main_config.policy.model.world_model_cfg
        wm.evaluator_env_num = n_eval
        wm.env_num = max(main_config.env.collector_env_num, n_eval)
    main_config.env.stop_value = int(1e9)
    evaluate_every_env_steps(args.eval_every)

    exp_name = args.exp_name or f"{args.algo}_{args.mode}_seed{args.seed}"
    main_config.exp_name = os.path.join(args.root, exp_name)
    # Written next to the run folder first (LightZero may still rename the run folder), moved in at the end.
    os.makedirs(args.root, exist_ok=True)
    progress_tmp = os.path.join(args.root, f".{exp_name}_{os.getpid()}_progress.jsonl")
    record_progress(progress_tmp)

    t0 = time.time()
    policy = train_fn([main_config, create_config], seed=args.seed, model_path=args.resume, max_env_step=args.steps)
    elapsed = time.time() - t0

    # LightZero appends a timestamp to exp_name if the folder already existed; take the newest match.
    import glob
    candidates = [d for d in glob.glob(main_config.exp_name + "*") if os.path.isdir(d)]
    run_dir = max(candidates, key=os.path.getmtime) if candidates else main_config.exp_name
    if policy is not None:
        import torch
        os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
        torch.save(policy.learn_mode.state_dict(), os.path.join(run_dir, "ckpt", "final.pth.tar"))
    if os.path.exists(progress_tmp):
        os.replace(progress_tmp, os.path.join(run_dir, "progress.jsonl"))
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(dict(vars(args), exp_name=exp_name, run_dir=run_dir, wall_time_sec=round(elapsed, 1)), f, indent=2)
    print(f"[train_connect4] {args.algo} done in {elapsed / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
