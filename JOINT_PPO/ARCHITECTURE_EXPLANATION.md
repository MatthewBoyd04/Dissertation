# JOINT PPO - Corrected Architecture Explanation

## What is Joint PPO?

**Joint PPO** is a centralized training and centralized execution approach where:

- **Single Policy**: One neural network makes decisions for ALL agents
- **Centralized Observations**: All agent observations are concatenated
- **Centralized Rewards**: Individual agent rewards are aggregated (summed)
- **Centralized Actions**: One action vector for all agents, decoded per-agent

This differs from MAPPO where each agent has its own policy (decentralized execution) but they share a centralized critic for training.

---

## How the Fixed Architecture Works

### Training Phase

```
┌──────────────────────────────────────────────────────────┐
│  Gymnasium Vectorized Environment (8 parallel envs)     │
│  Each containing: GridWorld × 8 drones                   │
└──────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│  JointAgentWrapper × 8 (one per vectorized env)         │
│  Responsibilities:                                       │
│  - Concatenate observations from 8 drones               │
│  - Sum rewards across all drones                        │
│  - Decode single action vector into per-drone actions   │
│  - Handle agent termination (padding with zeros)        │
└──────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│  VecNormalize Wrapper                                    │
│  Normalizes observations (mean=0, std=1)                │
│  Normalizes rewards                                      │
│  Returns: obs_shape = (8, 5, 5, 6) for 8 drones        │
└──────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│  PPO Algorithm ("MlpPolicy")                             │
│  - Input: Flat concatenated observation vector          │
│  - Output: Action vector [drone1_action, ..., drone8_action]
│  - Sees all agent observations simultaneously           │
│  - Learns to coordinate through single policy           │
└──────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│  Trained Model saved:                                    │
│  - joint_ppo_model_8drones.zip (weights)               │
│  - joint_ppo_vec_normalize_8drones.pkl (stats)         │
└──────────────────────────────────────────────────────────┘
```

### Inference Phase (Simulation)

```
┌──────────────────────────────────────────────────────────┐
│  Fresh GridWorld Environment (1 env × 8 drones)        │
└──────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│  JointAgentWrapper                                       │
│  Concatenates observations and decodes actions          │
└──────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│  Load Model:                                             │
│  - PPO.load("model.zip")                                │
│  - Load VecNormalize stats (to normalize obs)           │
│  - Deterministic=False for exploration                 │
└──────────────────────────────────────────────────────────┘
           ↓
┌──────────────────────────────────────────────────────────┐
│  Per Step:                                               │
│  1. Get joint_obs from wrapper.reset()                  │
│  2. Normalize obs with loaded VecNormalize stats        │
│  3. Policy predicts action vector                       │
│  4. Wrapper decodes into per-drone actions              │
│  5. Environment steps with per-drone actions            │
│  6. Get reward and next observation                     │
└──────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why Concatenate Instead of Stacking?

```python
# Flatten approach (what we use)
[drone1_obs_flat, drone2_obs_flat, ..., drone8_obs_flat]
→ Single vector: shape (D*N,) where D=obs_dims, N=num_drones

# Alternative (doesn't work with PPO easily)
# Stack would preserve 3D structure but PPO typically expects flat input
```

### 2. Why Always Use possible_agents?

```python
# Correct: Consistent size even when agents terminate
obs = [drone_1_obs, drone_2_obs, ..., drone_8_obs]  # Always 8 entries
      # When drone_3 terminates: [obs1, obs2, zeros, obs4, ..., obs8]

# Wrong (what I found): Size changes
obs = [drone_1_obs, drone_2_obs, drone_4_obs, ...]  # Now only 7 entries!
      # PPO freaks out: "observation changed shape!"
```

### 3. Why Joint Reward (Sum)?

```python
# Joint approach: One reward signal
joint_reward = sum([r1, r2, r3, r4, r5, r6, r7, r8])
# Encourages team exploration and discovery

# Alternative: Individual rewards (would need separate policies)
# Doesn't fit centralized execution paradigm
```

### 4. Why VecNormalize Persistence?

```
Training observations:
- Raw: range [-1, 1000] with huge variance
- Normalized: mean=0, std=1 (smooth training)

Simulation without normalization:
- Model receives unnormalized obs
- Weights trained on normalized scale
- Predictions wildly off!

Solution:
- Save running mean/std during training
- Apply same normalization during inference
```

---

## Observation Space Example

**Per-Drone Observation** (from environment):

- Shape: (6, 5, 5) for visionRange=2
- Channels: terrain, discovered, self, others, reward, memory

**Joint Observation** (to PPO):

```
Drone_1: [6*5*5=150 values]
Drone_2: [150 values]
Drone_3: [150 values]
...
Drone_8: [150 values]
─────────────────────
Total: 1200 values in single vector
```

**Action Space** (from PPO):

```
Output: [action1, action2, ..., action8]
Each action ∈ {0=Up, 1=Right, 2=Down, 3=Left, 4=Stay}
Type: np.ndarray or np.int64 array
```

---

## Why the Previous Implementation Failed

### The MAPPO Problem

```python
# What was tried:
from MAPPO.centralized_critic import MAPPOPolicy

model = PPO(
    MAPPOPolicy,  # Design: Multi-agent with N separate actors
    env,          # But env is already flattened to single agent!
    policy_kwargs={"num_agents": 8}  # Tries to create 8 internal actors
)

# Result: Architecture mismatch
# ├─ Policy expects: obs_size = per_agent_obs
# └─ Receives: obs_size = per_agent_obs * 8 (all concatenated)
#    CRASH or terrible learning
```

### The VecNormalize Problem

```python
# Training with normalization:
for step in range(10_000_000):
    obs = env.reset()  # Returns normalized obs
    action = model.predict(obs)  # Model trained on normalized obs

# Simulation without loading normalization:
for episode in range(100):
    obs = env.reset()  # Returns raw (non-normalized) obs
    action = model.predict(obs)  # Model expects normalized!
    # Observations are 1000x the expected scale!
    # Policy output completely wrong
```

### The Observation Consistency Problem

```python
# Episode starts with all 8 drones
obs, _ = env.step(actions)
obs_flat = concat(obs)  # Size: 1200

# Drone 3 finds reward and terminates
obs, _, terminations, _, _ = env.step(actions)
# env.agents = [drone_1, drone_2, drone_4, ..., drone_8]  # 7 agents!
obs_flat = concat(obs)  # Size: 1050 (7 * 150)

# PPO sees: "Shape changed from 1200 to 1050!"
# ValueError or undefined behavior
```

---

## Validation

The fixes ensure:

✅ **Single unified policy** controls all agents
✅ **Consistent observation size** (1200 values) throughout episode
✅ **Observation normalization** applied consistently train→sim
✅ **Deterministic decoding** from action vector to per-agent actions
✅ **Fresh environment** for each simulation (no state leakage)
✅ **Proper error handling** with clear messages

---

## Running the Fixed Algorithm

### Training

```python
python Main_JOINT_PPO.py
# Creates:
# - joint_ppo_model_8drones.zip
# - joint_ppo_vec_normalize_8drones.pkl
```

### Simulation/Inference

```python
from simulate_JOINT_PPO import runSimulations
runSimulations(simulations=100, num_drones=8)
# Automatically loads both model and normalization stats
```

---

## Expected Behavior

1. **First 500k steps**: Single 15×15 map (learning to navigate simply)
2. **500k-2.5M steps**: Mix of 15×15 and 30×30 maps (curriculum learning)
3. **2.5M+ steps**: All map sizes, increasingly harder environments

**Exploration improvements** should show in CSV logs:

- Reward discovery % increases
- Average steps to reward decreases
- Tiles discovered per step increases
