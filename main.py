import numpy as np
import time
from Environment.environment import GridWorldEnvironment
from Environment import Maps  # your preset map file

# ---------------------------------------------
# Q-Learning parameters
# ---------------------------------------------
alpha = 0.1          # Learning rate
gamma = 0.95         # Discount factor
epsilon = 0.1        # Exploration rate
episodes = 1000       # Number of episodes
max_steps = 200      # Max steps per episode
render_every = 100    # Render every X episodes
render_training = True  # Set to True to render during training, False for evaluation only

# ---------------------------------------------
# Initialize environment
# ---------------------------------------------
env = GridWorldEnvironment(grid_size=9, preset_map=Maps.PRESET_MAP_2)
n_actions = env.action_space.n
grid_size = env.grid_size

# Q-table: one value per (row, col, action)
Q = np.zeros((grid_size, grid_size, n_actions))

# ---------------------------------------------
# Training loop
# ---------------------------------------------
for episode in range(episodes):
    obs, info = env.reset()
    state = tuple(env._GridWorldEnvironment__agent_location)  # (row, col)
    total_reward = 0
    
    # Show episode counter
    if render_training and episode % render_every == 0:
        print(f"\n--- Episode {episode + 1}/{episodes} ---")

    for step in range(max_steps):
        # Only render every X episodes if training rendering is enabled
        if render_training and episode % render_every == 0:
            env.render()
            time.sleep(0.01)
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state[0], state[1], :])

        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = tuple(env._GridWorldEnvironment__agent_location)

        # Q-learning update
        best_next_action = np.argmax(Q[next_state[0], next_state[1], :])
        td_target = reward + gamma * Q[next_state[0], next_state[1], best_next_action]
        td_error = td_target - Q[state[0], state[1], action]
        Q[state[0], state[1], action] += alpha * td_error

        state = next_state
        total_reward += reward

        if terminated:
            break

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}/{episodes} - Total reward: {total_reward:.2f}")

# ---------------------------------------------
# Evaluation (watch the trained agent)
# ---------------------------------------------
print("\nTraining complete! Running one evaluation episode...\n")
obs, info = env.reset()
state = tuple(env._GridWorldEnvironment__agent_location)

for step in range(50):
    env.render()
    time.sleep(0.05)

    action = np.argmax(Q[state[0], state[1], :])
    next_obs, reward, terminated, truncated, info = env.step(action)
    state = tuple(env._GridWorldEnvironment__agent_location)

    if terminated:
        print("✅ Agent reached the goal!")
        break

env.close()