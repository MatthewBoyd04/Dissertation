import numpy as np
import gymnasium as gym
from graphics import GraphWin, Rectangle, Point
from . import Maps
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LoggerConfig import log


class MultiAgentGridWorldEnvironment(gym.Env):
    """Multi-agent grid world environment with 2 agents."""
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, grid_size: int = 9, preset_map=None, render_mode: str = "human", num_rewards: int = 3):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.tile_size = 30
        self.window = None
        self.num_rewards = num_rewards
        
        # Two agents starting at opposite corners
        self.agent_locations = [
            np.array([0, 0]),  # Agent 1: top-left
            np.array([0, grid_size - 1])  # Agent 2: top-right
        ]
        
        # Load map
        if preset_map is not None:
            self._load_map(preset_map)
        else:
            self._load_map(Maps.PRESET_MAP_1)
        
        # Update agent 2 position based on actual grid size
        self.agent_locations[1] = np.array([0, self.grid_size - 1])
        
        # Place rewards at strategic locations
        self.target_locations = [
            np.array([self.grid_size - 1, self.grid_size - 1]),  # Bottom-right
            np.array([self.grid_size - 1, 0]),  # Bottom-left
            np.array([self.grid_size // 2, self.grid_size // 2])  # Center
        ]
        
        # Track which targets are still active
        self.targets_active = [True] * self.num_rewards
        
        self.vision_range = 4
        obs_size = self.vision_range * 2 + 1
        
        # Observation space for each agent
        self.observation_space = gym.spaces.Box(
            low=0, high=4, shape=(obs_size, obs_size), dtype=np.int32
        )
        self.action_space = gym.spaces.Discrete(4)
        
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1])    # Right
        }
        
        self.grid_rectangles = None
        self.last_agent_positions = [None, None]
        
        # For observation rendering
        self.obs_windows = [None, None]
        self.obs_rectangles = [{}, {}]
        self.last_obs = [None, None]
        
    def _load_map(self, preset_map):
        """Load obstacle layout from preset map."""
        if preset_map.shape[0] != preset_map.shape[1]:
            raise ValueError(f"Preset map must be square, got shape {preset_map.shape}")
        
        self.grid_size = preset_map.shape[0]
        self.obstacles = preset_map == 3
        
    def _get_obs(self, agent_id):
        """Get observation for specific agent."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        grid[self.obstacles] = 3  # Obstacles
        
        # Mark all active targets
        for i, active in enumerate(self.targets_active):
            if active:
                grid[tuple(self.target_locations[i])] = 2
        
        # Mark agents (current agent as 1, other agent as 4)
        for i, pos in enumerate(self.agent_locations):
            if i == agent_id:
                grid[tuple(pos)] = 1  # Current agent
            else:
                grid[tuple(pos)] = 4  # Other agent
        
        # Extract vision window
        r, c = self.agent_locations[agent_id]
        vr = self.vision_range
        r_min, r_max = max(0, r - vr), min(self.grid_size, r + vr + 1)
        c_min, c_max = max(0, c - vr), min(self.grid_size, c + vr + 1)
        
        obs = np.zeros((vr * 2 + 1, vr * 2 + 1), dtype=np.int32)
        visible = grid[r_min:r_max, c_min:c_max]
        
        obs[
            (vr - (r - r_min)):(vr + (r_max - r)),
            (vr - (c - c_min)):(vr + (c_max - c))
        ] = visible
        
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        self.agent_locations = [
            np.array([0, 0]),  # Agent 1: top-left
            np.array([0, self.grid_size - 1])  # Agent 2: top-right
        ]
        
        # Reset targets
        self.targets_active = [True] * self.num_rewards
        
        observations = [self._get_obs(0), self._get_obs(1)]
        return observations, {}
    
    def step(self, actions):
        """Step with actions for both agents."""
        rewards = [0, 0]
        terminated = [False, False]
        
        # Move each agent
        for i, action in enumerate(actions):
            direction = self._action_to_direction[action]
            new_location = np.clip(
                self.agent_locations[i] + direction, 
                0, self.grid_size - 1
            )
            
            # Check for wall collision
            wall_collision = False
            if not self.obstacles[tuple(new_location)]:
                # Check for agent collision
                agent_collision = any(
                    np.array_equal(new_location, self.agent_locations[j]) 
                    for j in range(len(self.agent_locations)) if j != i
                )
                
                if not agent_collision:
                    self.agent_locations[i] = new_location
                #else:
                    #rewards[i] = -0.1  # Penalty for agent collision
            else:
                wall_collision = True
                rewards[i] = -0.1  # Penalty for wall collision
            
            # Check if agent reached any active target
            for j, target_active in enumerate(self.targets_active):
                if target_active and np.array_equal(self.agent_locations[i], self.target_locations[j]):
                    self.targets_active[j] = False  # Deactivate target
                    rewards[i] = 1.0
                    break
            
            if not wall_collision and rewards[i] == 0:
                rewards[i] = -0.01  # Small step penalty
        
        observations = [self._get_obs(0), self._get_obs(1)]
        return observations, rewards, terminated, [False, False], {}
    
    def _get_obs_with_bounds(self, agent_id):
        """Return observation with out-of-bounds areas marked as -1."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        grid[self.obstacles] = 3
        
        # Mark all active targets
        for i, active in enumerate(self.targets_active):
            if active:
                grid[tuple(self.target_locations[i])] = 2
        
        # Mark agents
        for i, pos in enumerate(self.agent_locations):
            if i == agent_id:
                grid[tuple(pos)] = 1
            else:
                grid[tuple(pos)] = 4
        
        r, c = self.agent_locations[agent_id]
        vr = self.vision_range
        obs = np.full((vr * 2 + 1, vr * 2 + 1), -1, dtype=np.int32)
        
        for obs_r in range(vr * 2 + 1):
            for obs_c in range(vr * 2 + 1):
                grid_r = r - vr + obs_r
                grid_c = c - vr + obs_c
                
                if 0 <= grid_r < self.grid_size and 0 <= grid_c < self.grid_size:
                    obs[obs_r, obs_c] = grid[grid_r, grid_c]
        
        return obs
    
    def render(self, show_observation=True):
        """Render the environment."""
        if self.render_mode != "human":
            return
        
        if self.window is None:
            win_size = self.grid_size * self.tile_size
            self.window = GraphWin("Multi-Agent GridWorld", win_size, win_size)
            self.grid_rectangles = {}
            
            # Draw grid
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    x1, y1 = c * self.tile_size, r * self.tile_size
                    x2, y2 = x1 + self.tile_size, y1 + self.tile_size
                    rect = Rectangle(Point(x1, y1), Point(x2, y2))
                    
                    if self.obstacles[r, c]:
                        rect.setFill("black")
                    else:
                        rect.setFill("white")
                    
                    rect.draw(self.window)
                    self.grid_rectangles[(r, c)] = rect
            
            # Draw active targets
            target_colors = ["green", "lightgreen", "yellow"]
            for i, active in enumerate(self.targets_active):
                if active:
                    self.grid_rectangles[tuple(self.target_locations[i])].setFill(target_colors[i])
            
            # Draw agents
            self.grid_rectangles[tuple(self.agent_locations[0])].setFill("blue")
            self.grid_rectangles[tuple(self.agent_locations[1])].setFill("red")
            
            self.last_agent_positions = [
                tuple(self.agent_locations[0]), 
                tuple(self.agent_locations[1])
            ]
        else:
            # Update agent positions
            colors = ["blue", "red"]
            target_colors = ["green", "lightgreen"]
            
            for i, (current_pos, last_pos) in enumerate(
                zip([tuple(pos) for pos in self.agent_locations], self.last_agent_positions)
            ):
                if current_pos != last_pos:
                    # Restore old position
                    if last_pos:
                        old_r, old_c = last_pos
                        if self.obstacles[old_r, old_c]:
                            self.grid_rectangles[last_pos].setFill("black")
                        else:
                            # Check if old position was an active target
                            target_colors = ["green", "lightgreen", "yellow"]
                            restored = False
                            for j, active in enumerate(self.targets_active):
                                if active and last_pos == tuple(self.target_locations[j]):
                                    self.grid_rectangles[last_pos].setFill(target_colors[j])
                                    restored = True
                                    break
                            if not restored:
                                self.grid_rectangles[last_pos].setFill("white")
                    
                    # Set new position
                    self.grid_rectangles[current_pos].setFill(colors[i])
                    self.last_agent_positions[i] = current_pos
        
        # Render observation windows if requested
        if show_observation:
            self._render_observations()
    
    def _render_observations(self):
        """Render both agents' observations in separate windows."""
        for agent_id in range(2):
            obs = self._get_obs_with_bounds(agent_id)
            obs_size = obs.shape[0]
            
            if self.obs_windows[agent_id] is None:
                win_size = obs_size * 60
                self.obs_windows[agent_id] = GraphWin(f"Agent {agent_id + 1} Observation", win_size, win_size)
                
                for r in range(obs_size):
                    for c in range(obs_size):
                        x1, y1 = c * 60, r * 60
                        x2, y2 = x1 + 60, y1 + 60
                        rect = Rectangle(Point(x1, y1), Point(x2, y2))
                        rect.draw(self.obs_windows[agent_id])
                        self.obs_rectangles[agent_id][(r, c)] = rect
                
                self.last_obs[agent_id] = np.full_like(obs, -1)
            
            # Update changed tiles
            for r in range(obs_size):
                for c in range(obs_size):
                    if self.last_obs[agent_id] is None or self.last_obs[agent_id][r, c] != obs[r, c]:
                        rect = self.obs_rectangles[agent_id][(r, c)]
                        
                        if obs[r, c] == -1:
                            rect.setFill("grey")
                        elif obs[r, c] == 0:
                            rect.setFill("white")
                        elif obs[r, c] == 1:
                            rect.setFill("blue" if agent_id == 0 else "red")
                        elif obs[r, c] == 2:
                            rect.setFill("green")
                        elif obs[r, c] == 3:
                            rect.setFill("black")
                        elif obs[r, c] == 4:
                            rect.setFill("red" if agent_id == 0 else "blue")
            
            self.last_obs[agent_id] = obs.copy()
    
    def close(self):
        """Close all windows."""
        if self.window is not None:
            self.window.close()
            self.window = None
        for i in range(2):
            if self.obs_windows[i] is not None:
                self.obs_windows[i].close()
                self.obs_windows[i] = None