import argparse
import csv
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from agents.rl_agent import ACTIONS, QLearningPolicy, default_model_path
from server.logic import Frogger

PolicyFn = Callable[[Dict[str, Any]], str]


def run_episode(policy_fn: PolicyFn, max_steps: int = 350, warmup_frames: int = 0) -> Dict[str, Any]:
    game = Frogger()
    reached_mid_checkpoint = False
    completed_lap = False

    for _ in range(warmup_frames):
        game.update(1.0 / 30.0)

    for step in range(max_steps):
        previous_checkpoint = game.current_lap_checkpoint
        previous_laps = game.laps
        state = game.get_state()
        action = policy_fn(state)
        game.move_frog(action, ignore_cooldown=False)

        for _ in range(6):
            game.update(1.0 / 30.0)
            if game.game_over:
                break

        reached_mid_checkpoint = reached_mid_checkpoint or (
            previous_checkpoint == 0 and game.current_lap_checkpoint == 50
        )
        completed_lap = completed_lap or game.laps > previous_laps

        if game.game_over:
            break

    return {
        "score": game.score,
        "high_score": game.high_score,
        "lives": game.lives,
        "laps": game.laps,
        "reached_mid_checkpoint": reached_mid_checkpoint,
        "completed_lap": completed_lap,
        "warmup_frames": warmup_frames,
        "steps": step + 1,
    }


def evaluate(
    name: str,
    policy_fn: PolicyFn,
    episodes: int,
    max_warmup_frames: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    logs: List[Dict[str, Any]] = []
    for episode in range(episodes):
        warmup_frames = random.randint(0, max_warmup_frames) if max_warmup_frames > 0 else 0
        result = run_episode(policy_fn, warmup_frames=warmup_frames)
        result["agent"] = name
        result["episode"] = episode + 1
        logs.append(result)

    scores = [row["score"] for row in logs]
    high_scores = [row["high_score"] for row in logs]
    reached = [row["reached_mid_checkpoint"] for row in logs]
    completed = [row["completed_lap"] for row in logs]
    return (
        {
            "agent": name,
            "episodes": episodes,
            "max_warmup_frames": max_warmup_frames,
            "average_score": round(sum(scores) / max(episodes, 1), 2),
            "best_score": max(scores) if scores else 0,
            "average_high_score": round(sum(high_scores) / max(episodes, 1), 2),
            "best_high_score": max(high_scores) if high_scores else 0,
            "mid_checkpoint_rate": round(100.0 * sum(reached) / max(episodes, 1), 2),
            "lap_completion_rate": round(100.0 * sum(completed) / max(episodes, 1), 2),
        },
        logs,
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SI2 Frogger agents offline.")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per agent.")
    parser.add_argument("--model", type=Path, default=default_model_path(), help="Q-table model path.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for repeatable evaluation.")
    parser.add_argument(
        "--max-warmup-frames",
        type=int,
        default=90,
        help="Maximum random traffic-only warmup frames before each episode. Use 0 for fixed-phase evaluation.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/evaluation_log.csv"), help="Evaluation CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    q_policy = QLearningPolicy.load(args.model, epsilon=0.0)
    q_summary, q_logs = evaluate(
        "Q-learning",
        lambda state: q_policy.choose_action(state) or "NORTH",
        args.episodes,
        args.max_warmup_frames,
    )
    random_summary, random_logs = evaluate(
        "Dummy random",
        lambda _state: random.choice(ACTIONS),
        args.episodes,
        args.max_warmup_frames,
    )

    write_csv(args.output, q_logs + random_logs)
    write_csv(args.output.parent / "evaluation_summary.csv", [q_summary, random_summary])

    for row in (q_summary, random_summary):
        print(
            f"{row['agent']}: avg={row['average_score']} best={row['best_score']} "
            f"avg_high={row['average_high_score']} best_high={row['best_high_score']} "
            f"mid_checkpoint={row['mid_checkpoint_rate']}% lap_completion={row['lap_completion_rate']}%"
        )


if __name__ == "__main__":
    main()
