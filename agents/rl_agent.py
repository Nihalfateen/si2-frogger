import argparse
import asyncio
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from agents.base_agent import BaseAgent as _BaseAgent
except ModuleNotFoundError:
    _BaseAgent = object

Action = str
StateKey = Tuple[int, int, int, int, int]

ACTIONS: Tuple[Action, ...] = ("NORTH", "EAST", "WEST", "SOUTH")
ROAD_LANES = {1, 2, 3, 5, 6, 7}
CHECKPOINT_LANES = {0, 4, 8}


class QLearningPolicy:
    """Small tabular Q-learning policy with a deterministic safety fallback."""

    def __init__(
        self,
        q_table: Optional[Dict[str, Dict[Action, float]]] = None,
        epsilon: float = 0.02,
        lookahead_seconds: float = 0.35,
    ) -> None:
        self.q_table = q_table or {}
        self.epsilon = epsilon
        self.lookahead_seconds = lookahead_seconds

    @classmethod
    def load(cls, model_path: Path, epsilon: float = 0.02) -> "QLearningPolicy":
        with model_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(q_table=payload.get("q_table", {}), epsilon=epsilon)

    def choose_action(self, state: Dict[str, Any], training: bool = False) -> Optional[Action]:
        if state.get("game_over"):
            return None

        legal_actions = self.legal_actions(state)
        if not legal_actions:
            return None

        safe_actions = [action for action in legal_actions if self.is_action_safe(state, action)]
        candidates = safe_actions or legal_actions

        if training and random.random() < self.epsilon:
            return random.choice(candidates)

        key = self.key_to_str(self.encode_state(state))
        learned_values = self.q_table.get(key, {})
        if learned_values:
            return max(
                candidates,
                key=lambda action: (
                    learned_values.get(action, 0.0),
                    self._action_priority(state, action),
                ),
            )

        return max(candidates, key=lambda action: self._action_priority(state, action))

    def legal_actions(self, state: Dict[str, Any]) -> List[Action]:
        x = int(round(float(state.get("frog_x", 0))))
        y = int(state.get("frog_y", 0))
        width = int(state.get("width", 11))
        height = int(state.get("height", 9))

        actions: List[Action] = []
        if y < height - 1:
            actions.append("NORTH")
        if x < width - 1:
            actions.append("EAST")
        if x > 0:
            actions.append("WEST")
        if y > 0:
            actions.append("SOUTH")
        return actions

    def encode_state(self, state: Dict[str, Any]) -> StateKey:
        x = int(round(float(state.get("frog_x", 0))))
        y = int(state.get("frog_y", 0))
        checkpoint_phase = 1 if y >= 4 else 0
        current_risk = self._risk_bucket(state, x, y)
        north_risk = self._risk_bucket(state, x, y + 1)
        lateral_bias = self._best_lateral_bias(state, x, y + 1)
        return (y, x, current_risk, north_risk, checkpoint_phase + lateral_bias)

    @staticmethod
    def key_to_str(key: StateKey) -> str:
        return "|".join(str(part) for part in key)

    @staticmethod
    def str_to_key(key: str) -> StateKey:
        return tuple(int(part) for part in key.split("|"))  # type: ignore[return-value]

    def is_action_safe(self, state: Dict[str, Any], action: Action) -> bool:
        x = int(round(float(state.get("frog_x", 0))))
        y = int(state.get("frog_y", 0))

        if action == "NORTH":
            y += 1
        elif action == "SOUTH":
            y -= 1
        elif action == "EAST":
            x += 1
        elif action == "WEST":
            x -= 1

        if y not in ROAD_LANES:
            return True

        return self._risk_bucket(state, x, y, lookahead=self.lookahead_seconds) == 0

    def _risk_bucket(
        self,
        state: Dict[str, Any],
        x: int,
        y: int,
        lookahead: Optional[float] = None,
    ) -> int:
        if y not in ROAD_LANES:
            return 0

        lookahead = self.lookahead_seconds if lookahead is None else lookahead
        width = int(state.get("width", 11))
        frog_left = x + 0.1
        frog_right = x + 0.9
        soon_left = x - 0.25
        soon_right = x + 1.25

        risk = 0
        for obstacle in state.get("obstacles", []):
            if int(obstacle.get("y", -1)) != y:
                continue
            obs_width = float(obstacle.get("width", 1.0))
            obs_x = float(obstacle.get("x", 0.0)) + float(obstacle.get("speed", 0.0)) * lookahead
            for left, right in self._wrapped_intervals(obs_x, obs_width, width):
                if left < frog_right and right > frog_left:
                    return 2
                if left < soon_right and right > soon_left:
                    risk = max(risk, 1)
        return risk

    @staticmethod
    def _wrapped_intervals(x: float, obstacle_width: float, board_width: int) -> Iterable[Tuple[float, float]]:
        start = math.fmod(x, board_width)
        if start < 0:
            start += board_width
        end = start + obstacle_width
        yield start, min(end, board_width)
        if end > board_width:
            yield 0.0, end - board_width

    def _best_lateral_bias(self, state: Dict[str, Any], x: int, y: int) -> int:
        if y not in ROAD_LANES:
            return 0
        left_risk = self._risk_bucket(state, max(0, x - 1), y)
        right_risk = self._risk_bucket(state, min(int(state.get("width", 11)) - 1, x + 1), y)
        if left_risk < right_risk:
            return -1
        if right_risk < left_risk:
            return 1
        return 0

    def _action_priority(self, state: Dict[str, Any], action: Action) -> int:
        y = int(state.get("frog_y", 0))
        if action == "NORTH":
            return 5
        if action in {"EAST", "WEST"}:
            return 3 if y in ROAD_LANES else 1
        return 0


class QLearningFroggerAgent(_BaseAgent):
    def __init__(self, server_uri: str, policy: QLearningPolicy) -> None:
        if _BaseAgent is object:
            raise RuntimeError("The 'websockets' dependency is required to run the online agent.")
        super().__init__(server_uri=server_uri)
        self.policy = policy

    async def deliberate(self) -> Optional[str]:
        if not self.current_state:
            return None
        return self.policy.choose_action(self.current_state)


def default_model_path() -> Path:
    project_root = find_project_root()
    return project_root / "models" / "q_table.json"


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "server").is_dir() and (parent / "agents").is_dir() and (parent / "README.md").exists():
            return parent
    return current.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SI2 Frogger Q-learning agent.")
    parser.add_argument("--server", default="ws://localhost:8765/ws", help="WebSocket server URI.")
    parser.add_argument("--model", type=Path, default=default_model_path(), help="Path to a trained q_table.json.")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Exploration rate used while running.")
    return parser.parse_args()


def build_policy(model_path: Path, epsilon: float) -> QLearningPolicy:
    if model_path.exists():
        return QLearningPolicy.load(model_path, epsilon=epsilon)
    return QLearningPolicy(epsilon=epsilon)


if __name__ == "__main__":
    args = parse_args()
    agent = QLearningFroggerAgent(args.server, build_policy(args.model, args.epsilon))
    asyncio.run(agent.run())
