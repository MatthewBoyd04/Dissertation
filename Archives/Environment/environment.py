
import sys
import time
import os
from typing import Optional
import numpy as np
import gymnasium as gym
from graphics import GraphWin, Rectangle, Point

# Import custom map module (must define PRESET_MAP_1)
from . import Maps

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LoggerConfig import log


class GridWorldEnvironment(gym.Env):
    """
    A simple 2D grid world environment where the agent must navigate
    to a target while avoiding obstacles.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size: int = 9, preset_map: Optional[np.ndarray] = None, render_mode: str = "human"):
        super().__init__()

        self.grid_size = grid_size  # Will be updated by _load_map if preset_map is provided
        self.render_mode = render_mode
        self.tile_size = 30  # Each tile size in pixels
        self.window = None

        # Agent and target initial positions
        self.__agent_location = np.array([0, 0])
        self.__target_location = np.array([grid_size - 1, grid_size - 1])  # Will be updated by _load_map
        
        # Load preset map or default one first to set correct grid size
        if preset_map is not None:
            self._load_map(preset_map)
        else:
            self._load_map(Maps.PRESET_MAP_1)
        
        log.d(f"Initialized GridWorldEnvironment with grid size: {self.grid_size}")

        # Let the agent see a (2*vision_range + 1) square window around itself
        self.vision_range = 2
        obs_size = self.vision_range * 2 + 1

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=3, shape=(obs_size, obs_size), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right

        # Action mapping
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1])    # Right
        }
        
        # For optimized rendering
        self.grid_rectangles = None
        self.last_agent_pos = None
        self.obs_window = None
        self.obs_rectangles = None
        self.last_obs = None



        # RNG (for reproducibility)
        self.np_random, _ = gym.utils.seeding.np_random(None)

    # ------------------------------------------------------------------
    # Internal utility methods
    # ------------------------------------------------------------------

    def _load_map(self, preset_map: np.ndarray):
        """Load obstacle layout from a pre-set numpy array."""
        if preset_map.shape[0] != preset_map.shape[1]:
            raise ValueError(f"Preset map must be square, got shape {preset_map.shape}")
        
        # Update grid size to match the map
        self.grid_size = preset_map.shape[0]
        self.__target_location = np.array([self.grid_size - 1, self.grid_size - 1])
        
        self.obstacles = preset_map == 3  # 3 represents barriers/walls
        log.d(f"Map loaded successfully with size {self.grid_size}x{self.grid_size}")

    def _get_obs(self) -> np.ndarray:
        """Return a (2*vision_range+1)x(2*vision_range+1) grid centered on the agent."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        grid[self.obstacles] = 3           # Mark obstacles
        grid[tuple(self.__target_location)] = 2
        grid[tuple(self.__agent_location)] = 1

        r, c = self.__agent_location
        vr = self.vision_range
        r_min, r_max = max(0, r - vr), min(self.grid_size, r + vr + 1)
        c_min, c_max = max(0, c - vr), min(self.grid_size, c + vr + 1)

        obs = np.zeros((vr * 2 + 1, vr * 2 + 1), dtype=np.int32)
        visible = grid[r_min:r_max, c_min:c_max]

        # Paste visible portion into observation grid
        obs[
            (vr - (r - r_min)):(vr + (r_max - r)),
            (vr - (c - c_min)):(vr + (c_max - c))
        ] = visible

        return obs
    
    def _get_obs_with_bounds(self) -> np.ndarray:
        """Return observation with out-of-bounds areas marked as -1."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        grid[self.obstacles] = 3
        grid[tuple(self.__target_location)] = 2
        grid[tuple(self.__agent_location)] = 1

        r, c = self.__agent_location
        vr = self.vision_range
        obs = np.full((vr * 2 + 1, vr * 2 + 1), -1, dtype=np.int32)  # Start with out-of-bounds
        
        # Only fill in valid grid positions
        for obs_r in range(vr * 2 + 1):
            for obs_c in range(vr * 2 + 1):
                grid_r = r - vr + obs_r
                grid_c = c - vr + obs_c
                
                if 0 <= grid_r < self.grid_size and 0 <= grid_c < self.grid_size:
                    obs[obs_r, obs_c] = grid[grid_r, grid_c]
        
        return obs

    # ------------------------------------------------------------------
    # Gym API methods
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        log.d("Environment reset called")

        self.__agent_location = np.array([0, 0])
        self.__target_location = np.array([self.grid_size - 1, self.grid_size - 1])

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        """Move the agent in the given direction, if possible."""
        log.d(f"Step called with action: {action}")

        direction = self._action_to_direction[action]
        new_location = np.clip(self.__agent_location + direction, 0, self.grid_size - 1)

        # Check for wall collision
        wall_collision = False
        if not self.obstacles[tuple(new_location)]:
            self.__agent_location = new_location
        else:
            log.d("Agent tried to move into a wall!")
            wall_collision = True

        # Check termination condition
        terminated = np.array_equal(self.__agent_location, self.__target_location)
        truncated = False
        
        # Reward structure
        if terminated:
            reward = 1.0
        elif wall_collision:
            reward = -0.1  # Negative reward for hitting walls
        else:
            reward = -0.01  # Small negative reward for each step

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, show_observation=True):
        """Render the grid world using graphics.py with optimized updates."""
        if self.render_mode != "human":
            return

        # Create window and initial grid if not already open
        if self.window is None:
            win_size = self.grid_size * self.tile_size
            self.window = GraphWin("GridWorld", win_size, win_size)
            self.grid_rectangles = {}
            
            # Draw static elements once
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    x1, y1 = c * self.tile_size, r * self.tile_size
                    x2, y2 = x1 + self.tile_size, y1 + self.tile_size
                    rect = Rectangle(Point(x1, y1), Point(x2, y2))
                    
                    if self.obstacles[r, c]:
                        rect.setFill("black")  # Wall
                    elif (r, c) == tuple(self.__target_location):
                        rect.setFill("green")  # Target
                    else:
                        rect.setFill("white")  # Empty
                    
                    rect.draw(self.window)
                    self.grid_rectangles[(r, c)] = rect
            
            # Draw agent initially
            agent_pos = tuple(self.__agent_location)
            self.grid_rectangles[agent_pos].setFill("blue")
            self.last_agent_pos = agent_pos
        else:
            # Only update agent position if it changed
            current_agent_pos = tuple(self.__agent_location)
            if self.last_agent_pos != current_agent_pos:
                # Restore old position
                old_r, old_c = self.last_agent_pos
                if self.obstacles[old_r, old_c]:
                    self.grid_rectangles[self.last_agent_pos].setFill("black")
                elif self.last_agent_pos == tuple(self.__target_location):
                    self.grid_rectangles[self.last_agent_pos].setFill("green")
                else:
                    self.grid_rectangles[self.last_agent_pos].setFill("white")
                
                # Set new position
                self.grid_rectangles[current_agent_pos].setFill("blue")
                self.last_agent_pos = current_agent_pos
        
        # Render observation window if requested
        if show_observation:
            self._render_observation()

    def _render_observation(self):
        """Render the agent's observation in a separate window with optimized updates."""
        obs = self._get_obs_with_bounds()
        obs_size = obs.shape[0]
        
        if self.obs_window is None:
            win_size = obs_size * 60  # Larger tiles for observation
            self.obs_window = GraphWin("Agent Observation", win_size, win_size)
            self.obs_rectangles = {}
            
            # Create all rectangles once
            for r in range(obs_size):
                for c in range(obs_size):
                    x1, y1 = c * 60, r * 60
                    x2, y2 = x1 + 60, y1 + 60
                    rect = Rectangle(Point(x1, y1), Point(x2, y2))
                    rect.draw(self.obs_window)
                    self.obs_rectangles[(r, c)] = rect
            
            self.last_obs = np.full_like(obs, -1)  # Initialize with invalid values
        
        # Only update changed tiles
        for r in range(obs_size):
            for c in range(obs_size):
                if self.last_obs[r, c] != obs[r, c]:
                    rect = self.obs_rectangles[(r, c)]
                    
                    if obs[r, c] == -1:
                        rect.setFill("grey")     # Out of bounds
                    elif obs[r, c] == 0:
                        rect.setFill("white")    # Empty
                    elif obs[r, c] == 1:
                        rect.setFill("blue")     # Agent
                    elif obs[r, c] == 2:
                        rect.setFill("green")    # Target
                    elif obs[r, c] == 3:
                        rect.setFill("black")    # Wall
        
        self.last_obs = obs.copy()
    
    def close(self):
        """Close the graphics windows."""
        if self.window is not None:
            self.window.close()
            self.window = None
        if self.obs_window is not None:
            self.obs_window.close()
            self.obs_window = None


# ----------------------------------------------------------------------
# Standalone testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = GridWorldEnvironment(grid_size=9, preset_map=Maps.PRESET_MAP_1)
    obs, info = env.reset()

    for step in range(10):
        env.render()
        time.sleep(0.5)
        action = np.random.choice(4)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            log.d("Reached the target!")
            break

    env.close()