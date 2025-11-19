import numpy as np
import time
from Environment.multi_agent_environment import MultiAgentGridWorldEnvironment
from Environment import Maps

# Q-Learning parameters
alpha = 0.1
gamma = 0.95
epsilon = 0.2
episodes = 10000
max_steps = 200
render_every = 1000
render_training = True

# Initialize multi-agent environment with 3 rewards
env = MultiAgentGridWorldEnvironment(grid_size=9, preset_map=Maps.PRESET_MAP_2, num_rewards=3)
n_actions = env.action_space.n
grid_size = env.grid_size

# Q-tables for both agents
Q_agents = [
    np.zeros((grid_size, grid_size, n_actions)),  # Agent 1
    np.zeros((grid_size, grid_size, n_actions))   # Agent 2
]

# Training loop
for episode in range(episodes):
    observations, info = env.reset()
    states = [tuple(env.agent_locations[0]), tuple(env.agent_locations[1])]
    total_rewards = [0, 0]
    
    if render_training and episode % render_every == 0:
        print(f"\n--- Episode {episode + 1}/{episodes} ---")

    for step in range(max_steps):
        if render_training and episode % render_every == 0:
            env.render()
            time.sleep(0.01)
        
        # Select actions for both agents
        actions = []
        for i in range(2):
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                action = np.argmax(Q_agents[i][states[i][0], states[i][1], :])
            actions.append(action)
        
        # Take step
        next_observations, rewards, terminated, truncated, info = env.step(actions)
        next_states = [tuple(env.agent_locations[0]), tuple(env.agent_locations[1])]
        
        # Q-learning update for both agents
        for i in range(2):
            best_next_action = np.argmax(Q_agents[i][next_states[i][0], next_states[i][1], :])
            td_target = rewards[i] + gamma * Q_agents[i][next_states[i][0], next_states[i][1], best_next_action]
            td_error = td_target - Q_agents[i][states[i][0], states[i][1], actions[i]]
            Q_agents[i][states[i][0], states[i][1], actions[i]] += alpha * td_error
            
            total_rewards[i] += rewards[i]
        
        states = next_states
        
        # Continue running - no termination
    if render_training and episode % render_every == 0:
        print(f"Episode {episode+1}/{episodes} - Agent 1 reward: {total_rewards[0]:.2f}, Agent 2 reward: {total_rewards[1]:.2f}")

# Evaluation
print("\nTraining complete! Running evaluation episode...\n")
observations, info = env.reset()
states = [tuple(env.agent_locations[0]), tuple(env.agent_locations[1])]
collected_rewards = [False, False]

for step in range(100):
    env.render()
    time.sleep(0.1)
    
    # Use trained policies (greedy)
    actions = []
    for i in range(2):
        action = np.argmax(Q_agents[i][states[i][0], states[i][1], :])
        actions.append(action)
    
    next_observations, rewards, terminated, truncated, info = env.step(actions)
    states = [tuple(env.agent_locations[0]), tuple(env.agent_locations[1])]
    
    # Check if any agent collected a reward this step
    for i, reward in enumerate(rewards):
        if reward == 1.0:
            print(f"✅ Agent {i + 1} collected a reward!")
    
    # Stop if all rewards collected
    if not any(env.targets_active):
        print("All rewards have been collected!")
        break

env.close()