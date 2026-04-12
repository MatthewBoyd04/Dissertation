# Reward Shaping Improvements - March 24, 2026

## Problem Analysis

Joint PPO was achieving only 56-70% reward find rate on 15x15 maps despite 97-98% tile exploration coverage. The issue was that agents could see the reward but couldn't navigate to it effectively.

## Root Causes Identified

### 1. **Extreme Reward Sparsity**

- **Reward found**: +10,000 (only when stepping exactly on reward tile)
- **No intermediate guidance** for getting closer to reward
- Agents had no incentive to move toward reward once they could see it

### 2. **Immediate Agent Termination on Reward Collection**

```python
if self.grid[x, y] == 3:
    rewards[agent] += self.rewardWeight["rewardFound"]  # +10,000
    terminations[agent] = True  # Agent removed immediately!
```

**Problem**: Agent that finds reward gets massive reward but is terminated, preventing learning from successful behavior.

### 3. **Reward Scale Explosion in Joint PPO**

- With 8 agents and reward averaging: +10,000 → +1,250 per agent
- Signal too weak for effective learning

## Solutions Implemented

### 1. **Distance-Based Reward Shaping** ✅

```python
def _calculate_distance_reward(self, agent, prev_distance):
    """Calculate reward based on change in distance to reward"""
    current_distance = abs(x - rx) + abs(y - ry)  # Manhattan distance
    distance_change = prev_distance - current_distance
    return distance_change * self.rewardWeight["rewardProximity"]  # +1.0 per step closer
```

**Benefits**:

- Agents get continuous feedback for moving toward reward
- +1.0 reward for each step closer, -1.0 for each step further
- Provides clear navigation guidance

### 2. **Reduced Reward Scale** ✅

```python
rewardWeight = {
    "rewardFound": 100.0,  # Reduced from 10,000
    "rewardProximity": 1.0,  # New distance-based reward
    # ... other rewards unchanged
}
```

**Benefits**:

- Prevents reward scale explosion
- More balanced reward distribution
- Better gradient flow in Joint PPO averaging

### 3. **Non-Terminating Reward Collection** ✅

```python
# Before: Episode ended immediately when reward found
if self.reward_found:
    for agent in self.agents:
        truncations[agent] = True  # ❌ Episode ends immediately

# After: Episode continues after reward collection (like IPPO/MAPPO)
if not np.any(self.grid == 3):
    for agent in self.agents:
        truncations[agent] = True  # ✅ Episode ends when no rewards remain
```

**Benefits:**

- Episodes continue after reward collection, allowing continued learning
- Matches IPPO/MAPPO behavior for consistent training
- Agents can learn from successful navigation patterns
- Prevents premature episode termination

- Agent that finds reward can continue exploring
- Other agents get more time to learn from the successful behavior
- Prevents premature episode termination

### 4. **Episode Termination on Reward Collection** ✅

```python
# Episode ends when reward is found by any agent
if self.reward_found:
    for agent in self.agents:
        truncations[agent] = True
```

**Benefits**:

- Episode length appropriate to task difficulty
- Clear success signal when reward is found

## Additional Issues Identified - March 24, 2026

### 5. **Extreme Reward Scale Imbalance** ❌

**Analysis**: Exploration rewards are 14x higher than reward-finding rewards!

- **Max exploration reward**: 179 tiles × (0.5 + 1.5) = **448.5** per episode
- **Reward finding reward**: 500.0 total
- **Ratio**: Only 0.9x more from exploration (vs 14.3x before)

**Previous rewards** (causing exploration prioritization):

```python
rewardWeight = {
    "tileDiscovered": 2.0,      # Too high
    "individualDiscovery": 6.0, # Too high
    "rewardFound": 100.0,       # Too low
    "rewardProximity": 1.0,     # Too weak
}
```

**Corrected rewards** (prioritizing reward finding):

```python
rewardWeight = {
    "tileDiscovered": 0.5,      # Reduced 4x
    "individualDiscovery": 1.5, # Reduced 4x
    "rewardFound": 500.0,       # Increased 5x
    "rewardProximity": 5.0,     # Increased 5x
}
```

**Benefits**:

- Finding reward is now 500.0 vs max exploration of ~450
- Proximity reward (5.0) is 10x stronger than tile discovery (0.5 + 1.5 = 2.0)
- Agents prioritize navigation to reward over pure exploration

### 15x15 Maps (Current: 56-70% success)

- **Target**: 85-95% success rate
- **Rationale**: Distance-based rewards provide clear navigation guidance

### 30x30 Maps (Current: 16-31% success)

- **Target**: 40-60% success rate
- **Rationale**: Better exploration + navigation guidance

### 45x45 Maps (Current: 4-11% success)

- **Target**: 15-30% success rate
- **Rationale**: Improved signal-to-noise ratio in rewards

## Technical Details

### Reward Structure Now:

- **Distance improvement**: +1.0 per step closer to reward
- **Tile discovery**: +2.0 + 6.0 = +8.0 per new tile
- **Reward collection**: +100.0 (shared success signal)
- **Hazard penalty**: -100.0 (strong avoidance signal)
- **Step penalty**: -0.005 (slight efficiency pressure)

### Joint PPO Compatibility:

- Distance rewards work well with reward averaging
- Non-terminating collection allows continued learning
- Balanced reward scales prevent gradient issues

## Testing Recommendations

1. **Short training runs** to verify reward signals are working
2. **Monitor average steps to reward** - should decrease as agents learn navigation
3. **Check reward distribution** - should see more consistent positive scores
4. **Compare against baseline** - same hyperparameters, different reward structure

This should bring Joint PPO performance much closer to IPPO and MAPPO levels for fair algorithm comparison.
