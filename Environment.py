# Needed for Environment
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np


#Helpers
from LoggerConfig import log
from graphics import GraphWin, Rectangle, Point
import time

#Testing Only
from Maps import map_15x15, map_30x30, map_45x45
from pettingzoo.test import parallel_api_test


class GridWorldEnvironment(ParallelEnv):

    metadata = {
        "name": "SAR_Environment",
        "render_modes": ["human"],
        }
    
    def __init__(
            self, 
            mapPreset: list[int], 
            agents:list[str], 
            maxCycles:int = 1000,
            visionRange: int = 2,
            render_every: int | None = None,
            tile_size: int = 16
        ):

        #Environment Variables
        self.grid = np.array(mapPreset)
        self.grid_size = len(mapPreset)
        self.maxCycles = maxCycles
        self.visionRange = visionRange
        self.step_count = 0
        
        log.d("Unique grid values:" + str(np.unique(self.grid)))

        #Rendering Variables 
        self.render_every = render_every
        self.tile_size = tile_size
        self.episode_count = 0
        self.render_enabled = False
        self.window = None
        self.static_drawn = False
        self.tile_rects = None
        self.agent_drawings = {}

        #Agent Variables
        self.possible_agents = agents
        self.agents = self.possible_agents[:]
        self.agent_positions = {} #Will be populated in the reset function, key is agent name, value is (x,y) position on the grid.

        #Action Space
        self.action_spaces = {
            agent: spaces.Discrete(5) # 0: Up, 1: Right, 2: Down, 3: Left, 4: Stay
            for agent in self.possible_agents
        }

        #Observation Space:
        obs_size = 2 * self.visionRange + 1
        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4, obs_size, obs_size),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        #Tracking Discovered Tiles
        self.discovered = np.zeros_like(self.grid, dtype=bool) #Keep track of discovered Tiles for analysis and rewards
    
    def reset(self, seed=None, options=None):
        self.episode_count += 1
        self.step_count = 0
        self.agents = self.possible_agents[:]
        self.discovered = np.zeros_like(self.grid, dtype=bool)
    
        # Rendering decision
        self.render_enabled = (
            self.render_every is not None and
            self.episode_count % self.render_every == 0
        )

        # If a new render episode starts, force full redraw
        if self.render_enabled:
            self.static_drawn = False

        #TO-DO - Add in the option to change starting position based on drone count, for now 4 drones is assumed. 
        
        # Initialize agent positions
        min_idx = 0
        mid_idx = (self.grid_size - 1) // 2
        max_idx = self.grid_size - 1
        self.agent_positions = {
            "Drone_1": [mid_idx, min_idx],
            "Drone_2": [min_idx, mid_idx],
            "Drone_3": [mid_idx, max_idx],
            "Drone_4": [max_idx, mid_idx]
        }

        # Mark initial discovered tiles
        for agent in self.agents:
            for x, y in self._visible_tiles(agent):
                self.discovered[x, y] = True

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}


        return observations, infos
    
    
    def step(self, actions):
        self.step_count += 1

        rewards = {agent: -0.1 for agent in self.agents}  # base step penalty
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.step_count >= self.maxCycles for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Move agents
        for agent, action in actions.items():
            self._move_agent(agent, action)

        # Discovery reward
        for agent in self.agents:
            newly_discovered = 0
            for x, y in self._visible_tiles(agent):
                if not self.discovered[x, y]:
                    self.discovered[x, y] = True
                    newly_discovered += 1
            rewards[agent] += newly_discovered * 0.1

        # Hazard penalty
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            if self.grid[x, y] == 2:
                terminations[agent] = True
                rewards[agent] = -1.0

        # Live agents
        live_agents = [
            agent for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        # Observations
        observations = self._get_obs(live_agents)

        # Update agent list
        self.agents = live_agents

        # Optional rendering
        if self.render_enabled:
            self.render()
            time.sleep(0.1)  # slow down visualization at the end
            if len(self.agents) == 0:
                self.close()
                

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
        size = 2 * self.visionRange + 1

        for agent in agents_list:
            obs = np.zeros((4, size, size), dtype=np.float32)

            x, y = self.agent_positions[agent]

            for dx in range(-self.visionRange, self.visionRange + 1):
                for dy in range(-self.visionRange, self.visionRange + 1):
                    nx, ny = x + dx, y + dy
                    obs_x, obs_y = dx + self.visionRange, dy + self.visionRange

                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        # Terrain
                        obs[0, obs_x, obs_y] = 1.0 if self.grid[nx, ny] in [1, 2] else 0.0
                        # Discovered tiles
                        obs[1, obs_x, obs_y] = 1.0 if self.discovered[nx, ny] else 0.0
                        # Self position
                        if nx == x and ny == y:
                            obs[2, obs_x, obs_y] = 1.0
                        # Other agents
                        for other, (ox, oy) in self.agent_positions.items():
                            if other != agent and ox == nx and oy == ny:
                                obs[3, obs_x, obs_y] = 1.0
                    else:
                        # Out-of-bounds padding (optional)
                        obs[:, obs_x, obs_y] = 0.0

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
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    # ---------------- Rendering ----------------
    def render(self):

        if not self.render_enabled:
            return

        grid_size = self.grid.shape[0]
        cell_size = 30
        win_size = grid_size * cell_size

        # -------------------------------------------------
        # Create window only once
        # -------------------------------------------------
        if self.window is None:
            self.window = GraphWin("GridWorld", win_size, win_size)
            self.window.setBackground("white")

        # -------------------------------------------------
        # Draw static grid only once
        # -------------------------------------------------
        if not self.static_drawn:

            self.tile_rects = [[None for _ in range(grid_size)] for _ in range(grid_size)]

            for i in range(grid_size):
                for j in range(grid_size):

                    x1 = j * cell_size
                    y1 = i * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size

                    rect = Rectangle(Point(x1, y1), Point(x2, y2))

                    if self.grid[i, j] == 1:
                        rect.setFill("black")
                    elif self.grid[i, j] == 2:
                        rect.setFill("red")
                    else:
                        rect.setFill("white")

                    rect.draw(self.window)
                    self.tile_rects[i][j] = rect

            self.static_drawn = True

        # -------------------------------------------------
        # Update discovered tiles only if changed
        # -------------------------------------------------
        for i in range(grid_size):
            for j in range(grid_size):

                if self.discovered[i, j] and self.grid[i, j] == 0:
                    self.tile_rects[i][j].setFill("lightblue")

        # -------------------------------------------------
        # Remove previous agents
        # -------------------------------------------------
        for drawing in self.agent_drawings.values():
            drawing.undraw()

        self.agent_drawings = {}

        # -------------------------------------------------
        # Draw agents
        # -------------------------------------------------
        for agent, (x, y) in self.agent_positions.items():

            x1 = y * cell_size
            y1 = x * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            rect = Rectangle(Point(x1, y1), Point(x2, y2))
            rect.setFill("green")
            rect.draw(self.window)

            self.agent_drawings[agent] = rect

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None

#Testing
if __name__ == "__main__":
    agents = ["Drone_1", "Drone_2", "Drone_3", "Drone_4"]

    env = GridWorldEnvironment(
        mapPreset=map_15x15,
        agents=agents,
        render_every=50  # render every 50 episodes
    )

    parallel_api_test(env)
