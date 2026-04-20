# Needed for Environment
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np


#Helpers
from LoggerConfig import log

#Testing Only
from Maps import map_15x15, map_30x30, map_45x45
from pettingzoo.test import parallel_api_test


class GridWorldEnvironment(ParallelEnv):

    metadata = {
        "name": "SAR_Environment",
        "render_modes": ["human"],
        }
    
    #Reward Weights:
    rewardWeight = {
        "tileDiscovered": 0.5,
        "rewardFound": 200,
        "HazardHit": -100,
        "Steps": -0.1,
        "approachReward": 5.0,  # per-step distance reduction toward visible reward tile
    }
    
    def __init__(
            self,
            mapPreset: list[int],
            num_drones: int = 4,
            maxCycles:int = 1000,
            visionRange: int = 2,
            use_map_memory: bool = False
        ):

        #Environment Variables
        self.grid = np.array(mapPreset)
        self.grid_size = len(mapPreset)
        self.maxCycles = maxCycles
        self.visionRange = visionRange
        self.step_count = 0
        self.use_map_memory = use_map_memory
        
        log.d("Unique grid values:" + str(np.unique(self.grid)))

        #Analysis Variables
        self.reward_found = False
        self.rewards_collected = 0
        self.reward_all_found_step = None
        self.hazard_terminations = 0
        self.has_hazards = bool(np.any(self.grid == 2))

        # Load training config (written by UI) to override reward weights and map mode
        import json as _json
        import os as _os
        _cfg_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "training_config.json")
        if _os.path.exists(_cfg_path):
            with open(_cfg_path) as _f:
                _cfg = _json.load(_f)
            for k, v in _cfg.get("reward_weights", {}).items():
                if k in self.rewardWeight:
                    self.rewardWeight[k] = v
            self.num_rewards = 3 if _cfg.get("map_mode") == "multiple_rewards" else 1
        else:
            self.num_rewards = 1

        #Agent Variables
        self.num_drones = max(1, min(num_drones, 8))  # Clamp between 1-8
        self.possible_agents = [f"Drone_{i+1}" for i in range(self.num_drones)]
        self.agents = self.possible_agents[:]
        self.agent_positions = {} #Will be populated in the reset function, key is agent name, value is (x,y) position on the grid.

        #Action Space
        self.action_spaces = {
            agent: spaces.Discrete(5) # 0: Up, 1: Right, 2: Down, 3: Left, 4: Stay
            for agent in self.possible_agents
        }

        #Observation Space:
        obs_size = 2 * self.visionRange + 1
        num_channels = 6 if use_map_memory else 5  # Add channel for map memory
        self.observation_spaces = {
            agent: spaces.Box(
                low=-1.0,  # -1 for undiscovered in memory
                high=1.0,
                shape=(num_channels, obs_size, obs_size),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        #Tracking Discovered Tiles
        self.discovered = np.zeros_like(self.grid, dtype=bool) #Keep track of discovered Tiles for analysis
        self.agent_discovered = {agent: np.zeros_like(self.grid, dtype=bool) for agent in self.possible_agents} #Track per-agent discoveries
        self.prev_reward_dists = {}
        
        # Map memory for each agent (-1 = undiscovered, 0 = available, 1 = blocked/hazard)
        if use_map_memory:
            self.agent_memory = {agent: np.full_like(self.grid, -1, dtype=np.float32) for agent in self.possible_agents}
    
    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.agents = self.possible_agents[:]
        self.discovered = np.zeros_like(self.grid, dtype=bool)
        self.agent_discovered = {agent: np.zeros_like(self.grid, dtype=bool) for agent in self.possible_agents}

        # Reset map memory
        if self.use_map_memory:
            self.agent_memory = {agent: np.full_like(self.grid, -1, dtype=np.float32) for agent in self.possible_agents}

        #Analysis Variables:
        self.reward_found = False
        self.rewards_collected = 0
        self.reward_all_found_step = None
        self.hazard_terminations = 0
        self.prev_reward_dists = {agent: None for agent in self.possible_agents}

        # Initialize agent positions based on drone count
        self.agent_positions = self._get_starting_positions()

        # Place random reward tile
        self._place_random_reward()

        # Mark initial discovered tiles (also mark per-agent to prevent rewarding start-visible tiles)
        for agent in self.agents:
            for x, y in self._visible_tiles(agent):
                self.discovered[x, y] = True
                self.agent_discovered[agent][x, y] = True

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}


        return observations, infos
    
    
    def step(self, actions):
        self.step_count += 1

        rewards = {agent: self.rewardWeight["Steps"] for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.step_count >= self.maxCycles for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Move agents
        for agent, action in actions.items():
            self._move_agent(agent, action)

        # Discovery reward - per agent (vectorised slice operations)
        for agent in self.agents:
            ax, ay = self.agent_positions[agent]
            x_lo = max(0, ax - self.visionRange);  x_hi = min(self.grid_size, ax + self.visionRange + 1)
            y_lo = max(0, ay - self.visionRange);  y_hi = min(self.grid_size, ay + self.visionRange + 1)

            self.discovered[x_lo:x_hi, y_lo:y_hi] = True

            view = self.agent_discovered[agent][x_lo:x_hi, y_lo:y_hi]
            newly_discovered = int(np.sum(~view))
            self.agent_discovered[agent][x_lo:x_hi, y_lo:y_hi] = True
            rewards[agent] += newly_discovered * self.rewardWeight["tileDiscovered"]

            if self.use_map_memory:
                w = self.grid[x_lo:x_hi, y_lo:y_hi]
                self.agent_memory[agent][x_lo:x_hi, y_lo:y_hi] = np.where(
                    (w == 1) | (w == 2), 1.0, 0.0)

        # Approach reward — dense shaping when reward tile is visible in window
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            r = self.visionRange
            x_lo = max(0, x - r); x_hi = min(self.grid_size, x + r + 1)
            y_lo = max(0, y - r); y_hi = min(self.grid_size, y + r + 1)
            window = self.grid[x_lo:x_hi, y_lo:y_hi]
            reward_coords = np.argwhere(window == 3)
            if len(reward_coords) > 0:
                abs_coords = reward_coords + np.array([x_lo, y_lo])
                dists = np.abs(abs_coords[:, 0] - x) + np.abs(abs_coords[:, 1] - y)
                current_dist = int(dists.min())
                prev = self.prev_reward_dists.get(agent)
                if prev is not None:
                    reduction = prev - current_dist
                    if reduction > 0:
                        rewards[agent] += self.rewardWeight["approachReward"] * reduction
                self.prev_reward_dists[agent] = current_dist
            else:
                self.prev_reward_dists[agent] = None

        # Reward tiles
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            if self.grid[x, y] == 3:
                rewards[agent] += self.rewardWeight["rewardFound"]
                terminations[agent] = True
                self.grid[x, y] = 0
                self.reward_found = True
                self.rewards_collected += 1
                if self.rewards_collected >= self.num_rewards:
                    self.reward_all_found_step = self.getStepsTaken()
                log.i(f"{agent} collected reward at ({x}, {y})")

        # Hazard penalty
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            if self.grid[x, y] == 2:
                terminations[agent] = True
                rewards[agent] += self.rewardWeight["HazardHit"]
                self.hazard_terminations += 1

        # Check if no rewards remain - truncate all agents
        if not np.any(self.grid == 3):
            for agent in self.agents:
                truncations[agent] = True
                self.reward_found = True

        # Live agents
        live_agents = [
            agent for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        # Observations
        observations = self._get_obs(live_agents)

        # Update agent list
        self.agents = live_agents

        return observations, rewards, terminations, truncations, infos
    
    def _move_agent(self, agent, action):
        x,y = self.agent_positions[agent]

        moves = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
            4: (0, 0)    # Stay
        }

        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy

        # Ensure the new position is within the grid boundaries
        if self._is_valid(new_x, new_y):
            self.agent_positions[agent] = [new_x, new_y]

    def _is_valid(self, x, y):
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return False
        
        if self.grid[x,y] == 1: #1 represents a blocked tile
            return False
        
        return True
        
    def _get_obs(self, agents_list=None):
        if agents_list is None:
            agents_list = self.agents

        observations = {}
        r = self.visionRange
        size = 2 * r + 1
        num_channels = 6 if self.use_map_memory else 5

        # Build occupancy grid once for all agents (O(1) lookup replaces O(N) inner loop)
        agent_pos_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for a, (ax, ay) in self.agent_positions.items():
            agent_pos_grid[ax, ay] = True

        for agent in agents_list:
            obs = np.zeros((num_channels, size, size), dtype=np.float32)
            x, y = self.agent_positions[agent]

            x_lo = max(0, x - r);  x_hi = min(self.grid_size, x + r + 1)
            y_lo = max(0, y - r);  y_hi = min(self.grid_size, y + r + 1)
            ox_lo = x_lo - x + r;  ox_hi = x_hi - x + r
            oy_lo = y_lo - y + r;  oy_hi = y_hi - y + r

            window = self.grid[x_lo:x_hi, y_lo:y_hi]
            obs[0, ox_lo:ox_hi, oy_lo:oy_hi] = (window == 1) | (window == 2)  # terrain
            obs[1, ox_lo:ox_hi, oy_lo:oy_hi] = self.discovered[x_lo:x_hi, y_lo:y_hi]
            obs[2, r, r] = 1.0                                                  # self (centre)
            obs[3, ox_lo:ox_hi, oy_lo:oy_hi] = agent_pos_grid[x_lo:x_hi, y_lo:y_hi]
            obs[3, r, r] = 0.0                                                  # exclude self
            obs[4, ox_lo:ox_hi, oy_lo:oy_hi] = (window == 3)                   # reward tiles
            if self.use_map_memory:
                obs[5, ox_lo:ox_hi, oy_lo:oy_hi] = self.agent_memory[agent][x_lo:x_hi, y_lo:y_hi]

            observations[agent] = obs

        return observations
    
    # ---------------- Visible tiles ----------------
    def _visible_tiles(self, agent):
        x, y = self.agent_positions[agent]
        tiles = []
        for dx in range(-self.visionRange, self.visionRange+1):
            for dy in range(-self.visionRange, self.visionRange+1):
                nx, ny = x+dx, y+dy
                if 0<=nx<self.grid_size and 0<=ny<self.grid_size:
                    tiles.append((nx, ny))
        return tiles
    
    def _get_starting_positions(self):
        mid = (self.grid_size - 1) // 2
        max_idx = self.grid_size - 1
        
        positions = [
            [mid, 0],           # 1: Left centre
            [mid, max_idx],     # 2: Right centre
            [0, mid],           # 3: Top centre
            [max_idx, mid],     # 4: Bottom centre
            [0, max_idx],       # 5: Top right
            [max_idx, max_idx], # 6: Bottom right
            [0, 0],             # 7: Top left
            [max_idx, 0]        # 8: Bottom left
        ]
        
        return {f"Drone_{i+1}": positions[i] for i in range(self.num_drones)}
    
    def _place_random_reward(self):
        # Clear existing rewards
        self.grid[self.grid == 3] = 0

        margin = max(2, self.grid_size // 8)  # 15x15→2, 30x30→3, 45x45→5
        valid_positions = []
        for x in range(margin, self.grid_size - margin):
            for y in range(margin, self.grid_size - margin):
                if self.grid[x, y] == 0:
                    valid_positions.append((x, y))

        count = min(getattr(self, "num_rewards", 1), len(valid_positions))
        chosen = [valid_positions[i] for i in np.random.choice(len(valid_positions), count, replace=False)]
        for rx, ry in chosen:
            self.grid[rx, ry] = 3
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def close(self):
        pass

    def getNumTilesDiscovered(self):
        return np.sum(self.discovered)
    
    def getStepsTaken(self):
        return self.step_count
    
    def getAnalysisScore(self):
        rewardFoundWeight = 50
        stepsTakenWeight = -1
        TilesDiscoveredWeight = 0.1

        score = (
            rewardFoundWeight * int(self.reward_found) +
            stepsTakenWeight * self.getStepsTaken() +
            TilesDiscoveredWeight * self.getNumTilesDiscovered()
            )
        
        return score
    
    def getEpisodeAnalysis(self):
        return {
            "reward_found": self.rewards_collected / self.num_rewards,
            "steps_taken": self.getStepsTaken(),
            "tiles_discovered": self.getNumTilesDiscovered(),
            "analysis_score": self.getAnalysisScore(),
            "Steps_to_find_reward_if_found": self.reward_all_found_step,
            "TilesDiscoveredPerStep": self.getNumTilesDiscovered()/self.getStepsTaken() if self.getStepsTaken() > 0 else 0,
            "hazard_terminations": self.hazard_terminations,
            "has_hazards": self.has_hazards,
        }

#Testing
if __name__ == "__main__":
    env = GridWorldEnvironment(
        mapPreset=map_15x15,
        num_drones=4,
    )

    parallel_api_test(env)
