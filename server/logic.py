from typing import Dict, List, Any

class Obstacle:
    def __init__(self, x: float, y: int, width: float, speed: float, type: str, variant: str = ""):
        self.x = x
        self.y = y
        self.width = width
        self.speed = speed
        self.type = type
        self.variant = variant

    def update(self, dt: float, grid_width: int):
        self.x += self.speed * dt
        # Use modulo-like wrap around to maintain spacing better
        if self.speed > 0 and self.x > grid_width:
            self.x -= (grid_width + self.width)
        elif self.speed < 0 and self.x < -self.width:
            self.x += (grid_width + self.width)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "speed": self.speed,
            "type": self.type,
            "variant": self.variant
        }

class Frogger:
    def __init__(self, width: int = 11, height: int = 9, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.high_score = 0
        self.reset_game()

    def reset_game(self):
        self.frog_x: float = float(self.width // 2)
        self.frog_y: int = 0
        self.lives = 3
        self.score = 0
        self.laps = 0
        self.current_lap_checkpoint = 0  # 0 or 50
        self.max_y_reached_in_checkpoint = 0 # Relative to last checkpoint
        self.game_over = False
        self.win = False # In infinite mode, win might not be used or used for high score
        self.obstacles: List[Obstacle] = []
        self._init_obstacles()
        
        # Frame-based movement cooldown (e.g., 6 frames at 30fps = 0.2s)
        self.move_cooldown_frames = 5 
        self.frames_since_last_move = self.move_cooldown_frames

    def _init_obstacles(self):
        self.obstacles = []
        
        # Reduced max speeds for "fast" variants
        variants = {
            "small_fast": (1.0, 1.8, "small_fast"),
            "small_slow": (1.0, 0.8, "small_slow"),
            "large_fast": (2.5, 1.5, "large_fast"),
            "large_slow": (2.5, 0.6, "large_slow")
        }

        # Lanes 1, 2, 3: Southbound
        self._add_lane(1, *variants["small_fast"], -1, 3)
        self._add_lane(2, *variants["large_slow"], -1, 2)
        self._add_lane(3, *variants["small_slow"], -1, 4)
        
        # Lanes 5, 6, 7: Northbound
        self._add_lane(5, *variants["large_fast"], 1, 2)
        self._add_lane(6, *variants["small_fast"], 1, 3)
        self._add_lane(7, *variants["large_slow"], 1, 2)

    def _add_lane(self, y: int, width: float, speed_mag: float, variant: str, direction: int, count: int):
        # Guarantee space
        max_count = int(self.width / (width + 1.5))
        actual_count = min(count, max_count)
        if actual_count < 1:
            actual_count = 1

        spacing = self.width / actual_count
        speed = speed_mag * direction
        for i in range(actual_count):
            self.obstacles.append(Obstacle(i * spacing, y, width, speed, "car", variant))

    def move_frog(self, direction: str, ignore_cooldown: bool = False):
        if self.game_over:
            return
            
        if not ignore_cooldown and self.frames_since_last_move < self.move_cooldown_frames:
            return

        moved = False
        if direction == "NORTH":
            if self.frog_y < self.height - 1:
                self.frog_y += 1
                self.frog_x = float(round(self.frog_x))
                moved = True
                
                # Scoring logic
                relative_y = self.frog_y - (4 if self.current_lap_checkpoint == 50 else 0)
                if relative_y > self.max_y_reached_in_checkpoint:
                    # Only add 10 points if we haven't reached a checkpoint lane yet
                    # Lane 4 and 8 are handled in _check_checkpoints
                    if self.frog_y != 4 and self.frog_y != 8:
                        self.score += 10
                        self.max_y_reached_in_checkpoint = relative_y
                        if self.score > self.high_score:
                            self.high_score = self.score

        elif direction == "SOUTH":
            if self.frog_y > 0:
                self.frog_y -= 1
                self.frog_x = float(round(self.frog_x))
                moved = True
        elif direction == "WEST":
            if self.frog_x > 0:
                self.frog_x -= 1
                moved = True
        elif direction == "EAST":
            if self.frog_x < self.width - 1:
                self.frog_x += 1
                moved = True

        if moved:
            self.frames_since_last_move = 0
            self._check_checkpoints()

    def _check_checkpoints(self):
        # Middle Checkpoint
        if self.frog_y == 4 and self.current_lap_checkpoint == 0:
            self.current_lap_checkpoint = 50
            self.score = self.laps * 100 + 50
            self.max_y_reached_in_checkpoint = 0
            if self.score > self.high_score:
                self.high_score = self.score
        
        # Final Checkpoint
        elif self.frog_y == 8:
            self.laps += 1
            self.current_lap_checkpoint = 0
            self.score = self.laps * 100
            self.max_y_reached_in_checkpoint = 0
            if self.score > self.high_score:
                self.high_score = self.score
            # Reset to start for infinite game
            self.frog_y = 0
            self.frog_x = float(self.width // 2)

    def _die(self):
        self.lives -= 1
        # Reset score to last checkpoint
        self.score = self.laps * 100 + self.current_lap_checkpoint
        # Move to last checkpoint
        self.frog_y = 4 if self.current_lap_checkpoint == 50 else 0
        self.frog_x = float(self.width // 2)
        self.max_y_reached_in_checkpoint = 0
        
        if self.lives <= 0:
            self.game_over = True

    def update(self, dt: float):
        self.frames_since_last_move += 1
        if self.game_over:
            return

        for obs in self.obstacles:
            obs.update(dt, self.width)

        # Check collisions in car lanes: 1-3 and 5-7 using vectorized numpy checks
        if (1 <= self.frog_y <= 3) or (5 <= self.frog_y <= 7):
            lane_obstacles = [obs for obs in self.obstacles if obs.y == self.frog_y]
            if lane_obstacles:
                import numpy as np
                obs_x = np.array([obs.x for obs in lane_obstacles])
                obs_w = np.array([obs.width for obs in lane_obstacles])
                
                frog_left = self.frog_x + 0.1
                frog_right = self.frog_x + 0.9
                
                # Check direct overlap
                overlap_normal = (obs_x < frog_right) & ((obs_x + obs_w) > frog_left)
                # Check wrap around right
                wrap_right_mask = (obs_x + obs_w) > self.width
                overlap_wrap_right = wrap_right_mask & (0 < frog_right) & ((obs_x + obs_w - self.width) > frog_left)
                # Check wrap around left
                wrap_left_mask = obs_x < 0
                overlap_wrap_left = wrap_left_mask & ((self.width + obs_x) < frog_right) & (self.width > frog_left)
                
                collisions = overlap_normal | overlap_wrap_right | overlap_wrap_left
                if np.any(collisions):
                    self._die()
                    return

    def get_state(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "frog_x": self.frog_x,
            "frog_y": self.frog_y,
            "lives": self.lives,
            "score": self.score,
            "high_score": self.high_score,
            "game_over": self.game_over,
            "win": self.win,
            "obstacles": [obs.to_dict() for obs in self.obstacles]
        }
