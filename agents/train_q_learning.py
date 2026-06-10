import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.rl_agent import ACTIONS, QLearningPolicy, default_model_path
from server.logic import Frogger


def run_training(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any], List[Dict[str, Any]]]:
    random.seed(seed)
    q_table: Dict[str, Dict[str, float]] = {}
    policy = QLearningPolicy(q_table=q_table, epsilon=epsilon)
    stats = {"episodes": episodes, "best_score": 0, "average_score": 0.0, "seed": seed}
    total_score = 0
    episode_logs: List[Dict[str, Any]] = []

    for episode in range(episodes):
        game = Frogger()
        previous_score = game.score
        previous_lives = game.lives
        episode_reward = 0.0
        stale_lane_steps = 0
        episode_reached_mid_checkpoint = False
        episode_completed_lap = False

        for step in range(350):
            state = game.get_state()
            previous_checkpoint = game.current_lap_checkpoint
            previous_laps = game.laps
            state_key = policy.key_to_str(policy.encode_state(state))
            q_table.setdefault(state_key, {action: 0.0 for action in ACTIONS})

            action = policy.choose_action(state, training=True) or "NORTH"
            game.move_frog(action, ignore_cooldown=True)

            for _ in range(6):
                game.update(1.0 / 30.0)
                if game.game_over:
                    break

            next_state = game.get_state()
            if next_state.get("frog_y", 0) > state.get("frog_y", 0):
                stale_lane_steps = 0
            else:
                stale_lane_steps += 1

            reached_mid_checkpoint = previous_checkpoint == 0 and game.current_lap_checkpoint == 50
            completed_lap = game.laps > previous_laps
            episode_reached_mid_checkpoint = episode_reached_mid_checkpoint or reached_mid_checkpoint
            episode_completed_lap = episode_completed_lap or completed_lap

            reward = reward_for_transition(
                previous_score,
                previous_lives,
                state,
                action,
                next_state,
                reached_mid_checkpoint=reached_mid_checkpoint,
                completed_lap=completed_lap,
                stale_lane_steps=stale_lane_steps,
            )
            episode_reward += reward

            next_key = policy.key_to_str(policy.encode_state(next_state))
            q_table.setdefault(next_key, {next_action: 0.0 for next_action in ACTIONS})
            best_next = max(q_table[next_key].values())
            old_value = q_table[state_key][action]
            q_table[state_key][action] = old_value + alpha * (reward + gamma * best_next - old_value)

            previous_score = game.score
            previous_lives = game.lives

            if game.game_over:
                break

        total_score += game.score
        stats["best_score"] = max(stats["best_score"], game.high_score, game.score)
        episode_logs.append(
            {
                "episode": episode + 1,
                "score": game.score,
                "high_score": game.high_score,
                "lives": game.lives,
                "steps": step + 1,
                "reward": round(episode_reward, 3),
                "reached_mid_checkpoint": episode_reached_mid_checkpoint,
                "completed_lap": episode_completed_lap,
                "epsilon": round(policy.epsilon, 5),
            }
        )

        # Slowly reduce exploration while preserving some late discovery.
        policy.epsilon = max(0.03, epsilon * (0.995 ** episode))

    stats["average_score"] = round(total_score / max(episodes, 1), 2)
    stats["states"] = len(q_table)
    return q_table, stats, episode_logs


def reward_for_transition(
    previous_score: int,
    previous_lives: int,
    state: Dict[str, Any],
    action: str,
    next_state: Dict[str, Any],
    reached_mid_checkpoint: bool = False,
    completed_lap: bool = False,
    stale_lane_steps: int = 0,
) -> float:
    reward = float(next_state.get("score", 0) - previous_score)

    if next_state.get("lives", previous_lives) < previous_lives:
        reward -= 150.0
    if reached_mid_checkpoint:
        reward += 50.0
    if completed_lap:
        reward += 100.0
    if next_state.get("frog_y", 0) > state.get("frog_y", 0):
        reward += 2.0
    if stale_lane_steps >= 5:
        reward -= 1.0
    if action == "SOUTH":
        reward -= 4.0
    if next_state.get("game_over"):
        reward -= 250.0

    return reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning policy for SI2 Frogger.")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.15, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=0.40, help="Initial exploration rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=Path, default=default_model_path(), help="Output model path.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for logs and plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_table, stats, episode_logs = run_training(args.episodes, args.alpha, args.gamma, args.epsilon, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    write_training_log(args.results_dir / "training_log.csv", episode_logs)
    write_training_curve(args.results_dir / "training_curve.png", episode_logs)

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "algorithm": "tabular_q_learning_with_safety_filter",
                "hyperparameters": {
                    "alpha": args.alpha,
                    "gamma": args.gamma,
                    "initial_epsilon": args.epsilon,
                },
                "stats": stats,
                "q_table": q_table,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(json.dumps(stats, indent=2, sort_keys=True))


def write_training_log(path: Path, episode_logs: List[Dict[str, Any]]) -> None:
    if not episode_logs:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_logs[0].keys()))
        writer.writeheader()
        writer.writerows(episode_logs)


def write_training_curve(path: Path, episode_logs: List[Dict[str, Any]], rolling_window: int = 25) -> None:
    if not episode_logs:
        return

    episodes = [row["episode"] for row in episode_logs]
    scores = [row["score"] for row in episode_logs]
    rolling_scores = []
    for index in range(len(scores)):
        window = scores[max(0, index - rolling_window + 1) : index + 1]
        rolling_scores.append(sum(window) / len(window))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, color="#84a7ff", linewidth=1.0, alpha=0.45, label="Episode score")
    plt.plot(episodes, rolling_scores, color="#143d8f", linewidth=2.2, label=f"{rolling_window}-episode rolling average")
    plt.title("SI2 Frogger Q-Learning Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
