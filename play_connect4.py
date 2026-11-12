"""Play Connect Four against your trained AlphaZero or MuZero agent in the terminal.

Usage (from this directory):
    source /venv/main/bin/activate
    python play_connect4.py                       # latest checkpoint (data_c4/ or experiments/), you move first
    python play_connect4.py --second              # the agent moves first
    python play_connect4.py --sims 200            # stronger (slower) agent search
    python play_connect4.py --ckpt data_c4/muzero_bot_seed0/ckpt/final.pth.tar
    python play_connect4.py --device cpu          # default: GPU if available
    python play_connect4.py --slip-prob 0.2       # stochastic: pieces may slip into a neighbouring column

The algorithm (alphazero / muzero) and training mode are read from the run's run_info.json.
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from connect4_agent import latest_checkpoint, load_agent

SYMBOLS = {0: "·", 1: "\033[91m●\033[0m", 2: "\033[93m●\033[0m"}  # empty, red, yellow


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
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--slip-prob", type=float, default=None,
                        help="stochastic env: probability a piece slips columns (default: as trained)")
    parser.add_argument("--slip-to", choices=("adjacent", "any"), default=None, help="default: as trained")
    parser.add_argument("--sim-slip-prob", type=float, default=None,
                        help="AlphaZero simulator slip (default: as trained / = --slip-prob)")
    args = parser.parse_args()

    agent = load_agent(args.ckpt or latest_checkpoint(), args.sims, args.device,
                       slip_prob=args.slip_prob, slip_to=args.slip_to, sim_slip_prob=args.sim_slip_prob)
    env = agent.make_env()
    obs = env.reset()

    human_player = 2 if args.second else 1
    print(f"Loaded {agent.ckpt}  [{agent.describe()}]")
    print(f"You are {SYMBOLS[human_player]}  (player {human_player}). Enter a column 1-7, or q to quit.")
    if env.slip_prob > 0:
        print(f"Slippery board: each piece lands in another ({env.slip_to}) column with probability {env.slip_prob:g}.")

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
            action = agent.act(env, obs)
            print(f"Agent plays column {action + 1}")
        timestep = env.step(action)
        obs, done = timestep.obs, timestep.done
        if env.last_slipped:
            print(f"  ...the piece slipped into column {env.last_landed_column + 1}!")

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
