import unittest
from unittest.mock import patch
from server.logic import Frogger
from agents.rl_agent import QLearningPolicy
from agents.evaluate_agents import run_episode
from agents.train_q_learning import parse_args

class TestFroggerLogic(unittest.TestCase):
    def setUp(self):
        self.game = Frogger(width=11, height=9)

    def test_initial_state(self):
        self.assertEqual(self.game.frog_x, 5)
        self.assertEqual(self.game.frog_y, 0)
        self.assertEqual(self.game.lives, 3)
        self.assertEqual(self.game.score, 0)
        self.assertFalse(self.game.game_over)

    def test_state_includes_training_and_runtime_metadata(self):
        state = self.game.get_state()

        self.assertEqual(state["laps"], 0)
        self.assertEqual(state["current_lap_checkpoint"], 0)
        self.assertEqual(state["frames_since_last_move"], self.game.move_cooldown_frames)
        self.assertEqual(state["move_cooldown_frames"], self.game.move_cooldown_frames)

    def test_move_north(self):
        self.game.move_frog("NORTH", ignore_cooldown=True)
        self.assertEqual(self.game.frog_y, 1)
        self.assertEqual(self.game.score, 10)

    def test_move_bounds(self):
        for _ in range(10):
            self.game.move_frog("WEST", ignore_cooldown=True)
        self.assertEqual(self.game.frog_x, 0)
        
        for _ in range(20):
            self.game.move_frog("EAST", ignore_cooldown=True)
        self.assertEqual(self.game.frog_x, 10)

    def test_road_collision(self):
        self.game.move_frog("NORTH", ignore_cooldown=True)
        self.game.obstacles = []
        from server.logic import Obstacle
        self.game.obstacles.append(Obstacle(x=5.0, y=1, width=2.5, speed=1.0, type="car", variant="large_fast"))
        self.game.update(0.1)
        self.assertEqual(self.game.lives, 2)
        # Score resets to checkpoint (0 since no goals/checkpoints)
        self.assertEqual(self.game.score, 0)
        self.assertEqual(self.game.frog_y, 0)

    def test_goal_reaching(self):
        # Move to Lane 8 (Final Checkpoint)
        for _ in range(8):
            self.game.move_frog("NORTH", ignore_cooldown=True)
        
        self.assertEqual(self.game.score, 100)
        self.assertEqual(self.game.frog_y, 0) # Resets to start
        self.assertEqual(self.game.laps, 1)

    def test_space_guarantee(self):
        self.game.obstacles = []
        self.game._add_lane(1, 2.5, 1.0, "variant", 1, 10)
        count = len([o for o in self.game.obstacles if o.y == 1])
        self.assertEqual(count, 2) 

    def test_score_checkpoint(self):
        # Reach middle checkpoint (50 points)
        for _ in range(4):
            self.game.move_frog("NORTH", ignore_cooldown=True)
        self.assertEqual(self.game.score, 50)
        
        # Advance and die
        self.game.move_frog("NORTH", ignore_cooldown=True) # Score = 60
        self.assertEqual(self.game.score, 60)
        self.game._die()
        
        # Score should reset to 50 (checkpoint), not 0
        self.assertEqual(self.game.score, 50)
        self.assertEqual(self.game.frog_y, 4)

    def test_high_score_tracking(self):
        self.game.move_frog("NORTH", ignore_cooldown=True) # Score = 10
        self.assertEqual(self.game.high_score, 10)
        
        self.game._die() # Score = 0, high_score = 10
        self.assertEqual(self.game.high_score, 10)
        
        self.game.move_frog("NORTH", ignore_cooldown=True) # Score = 10
        self.game.move_frog("NORTH", ignore_cooldown=True) # Score = 20
        self.assertEqual(self.game.high_score, 20)


class TestQLearningPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = QLearningPolicy(epsilon=0.0)
        self.game = Frogger(width=11, height=9)

    def test_policy_avoids_immediate_collision(self):
        self.game.frog_x = 5.0
        self.game.frog_y = 0
        from server.logic import Obstacle
        self.game.obstacles = [Obstacle(x=5.0, y=1, width=2.5, speed=0.0, type="car")]

        state = self.game.get_state()

        self.assertFalse(self.policy.is_action_safe(state, "NORTH"))
        self.assertNotEqual(self.policy.choose_action(state), "NORTH")

    def test_policy_prefers_progress_when_safe(self):
        self.game.obstacles = []
        state = self.game.get_state()

        self.assertEqual(self.policy.choose_action(state), "NORTH")


class TestEvaluationProcess(unittest.TestCase):
    def test_run_episode_records_warmup_and_progress_metadata(self):
        result = run_episode(lambda _state: "NORTH", max_steps=2, warmup_frames=7)

        self.assertEqual(result["warmup_frames"], 7)
        self.assertIn("high_score", result)
        self.assertIn("lives", result)
        self.assertIn("laps", result)
        self.assertIn("completed_lap", result)


class TestTrainingConfiguration(unittest.TestCase):
    def test_default_training_hyperparameters_match_checked_in_model(self):
        with patch("sys.argv", ["train_q_learning"]):
            args = parse_args()

        self.assertEqual(args.episodes, 3000)
        self.assertEqual(args.alpha, 0.15)
        self.assertEqual(args.gamma, 0.95)
        self.assertEqual(args.epsilon, 0.40)
        self.assertEqual(args.seed, 7)

if __name__ == "__main__":
    unittest.main()
