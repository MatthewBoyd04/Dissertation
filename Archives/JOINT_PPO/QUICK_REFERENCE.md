# JOINT PPO - Quick Reference: All Fixes

## Summary of Issues & Fixes (Complete List)

### ✅ Issue #1: Wrong Policy Class (CRITICAL)

**What was wrong:** Using `MAPPOPolicy` (multi-agent) for a centralized training algorithm
**Line changed:** `train_JOINT_PPO.py` line ~80
**Fix:** Changed to `"MlpPolicy"` (standard single-agent PPO)

### ✅ Issue #2: Missing VecNormalize Persistence

**What was wrong:** Training normalized observations but simulation didn't denormalize them
**Lines changed:**

- `train_JOINT_PPO.py` line 101: Added `env.save(vec_normalize_path)`
- `simulate_JOINT_PPO.py` lines 45-50: Added VecNormalize loading

### ✅ Issue #3: Inconsistent Observation Space

**What was wrong:** Observation vector size changed when agents terminated
**File changed:** `single_agent_wrapper.py` lines ~50-100
**Fix:** Always concatenate in `possible_agents` order, padding terminated agents with zeros

### ✅ Issue #4: Environment State Leaks

**What was wrong:** Same environment instance reused across simulations
**File:** `simulate_JOINT_PPO.py` lines ~80-95
**Fix:** Create fresh environment for each simulation

### ✅ Issue #5: Simplified Simulation Loop

**What was wrong:** Complex manual action decoding duplicating wrapper logic
**File:** `simulate_JOINT_PPO.py` lines ~75-110
**Fix:** Use wrapper's step() method directly for consistency

---

## NEW FIXES - March 2026 Update

### ✅ Fix #6: Improved Curriculum Learning (High Impact)

**What was wrong:** Curriculum jumped too aggressively from 15x15 to 30x30 before convergence
**File:** `train_JOINT_PPO.py` CurriculumWrapper.\_create_new_env()
**What changed:**

- 0-1M: 100% 15x15
- 1M-3M: 90% 15x15, 10% 30x30
- 3M-6M: 60% 15x15, 40% 30x30
- 6M-8M: 30% 15x15, 70% 30x30
- 8M-10M: 100% 30x30
- 10M+: 75% 30x30, 25% 45x45

### ✅ Fix #7: Disabled Reward Normalization (High Impact)

**What was wrong:** Normalizing rewards obscured reward signal structure
**File:** `train_JOINT_PPO.py` line ~105
**Change:** `VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)`
**Impact:** Clearer reward signals for learning

### ✅ Fix #8: Reward Averaging Instead of Sum (Medium Impact)

**What was wrong:** Summing rewards caused scale explosion when targets found
**Files:** `single_agent_wrapper.py` + `simulate_JOINT_PPO.py`
**What changed:** Added `use_reward_averaging=True` parameter
**Impact:** Using `np.mean(rewards)` instead of `sum(rewards)` prevents scale explosion

### ✅ Fix #9: Increased Batch Size (Medium Impact)

**What was wrong:** Batch size 256 too small for 1,200-dim observation space
**File:** `train_JOINT_PPO.py` line ~140
**Change:** `batch_size=512` (was 256), `n_steps=4096` (was 2048)
**Impact:** More stable gradient estimates

### ✅ Fix #10: Increased Entropy Coefficient (Medium Impact)

**What was wrong:** `ent_coef=0.005` discouraged exploration too much
**File:** `train_JOINT_PPO.py` line ~142
**Change:** `ent_coef=0.01` (doubled)
**Impact:** Better exploration, less premature convergence

---

## REWARD SHAPING FIXES - March 24, 2026 Update

### ✅ Fix #11: Distance-Based Reward Shaping (Major Impact)

**What was wrong:** Agents could see reward but had no guidance for navigating to it
**File:** `Environment.py` - Added `_calculate_distance_reward()` method
**What changed:**

- +1.0 reward per step closer to reward, -1.0 per step further
- Continuous navigation guidance instead of binary success/failure
- **Impact:** Should dramatically improve navigation to visible rewards

### ✅ Fix #12: Non-Terminating Reward Collection (Medium Impact)

**What was wrong:** Agent terminated immediately upon finding reward
**File:** `Environment.py` step() function
**What changed:** Agents get reward but continue exploring; episode ends when reward collected
**Impact:** Agents can learn from successful reward-finding behavior

### ✅ Fix #13: Balanced Reward Scales (Medium Impact)

**What was wrong:** Reward scale explosion (+10,000 → +100 for better Joint PPO compatibility)
**File:** `Environment.py` rewardWeight dict
**What changed:** rewardFound: 10,000 → 100, added rewardProximity: 1.0
**Impact:** Better gradient flow and reward signal balance

### ✅ Fix #14: Reward Scale Rebalancing (CRITICAL for Navigation)

**What was wrong:** Exploration rewards were 14x higher than reward-finding rewards, causing agents to prioritize exploration over finding the reward
**File:** `Environment.py` rewardWeight dict
**What changed:**

```python
# BEFORE (exploration prioritized):
"tileDiscovered": 2.0,      # Too high
"individualDiscovery": 6.0, # Too high
"rewardFound": 100.0,       # Too low
"rewardProximity": 1.0,     # Too weak

# AFTER (reward finding prioritized):
"tileDiscovered": 0.5,      # Reduced 4x
"individualDiscovery": 1.5, # Reduced 4x
"rewardFound": 500.0,       # Increased 5x
"rewardProximity": 5.0,     # Increased 5x
```

**Impact:** Finding reward (500.0) now outweighs max exploration reward (~450), with proximity rewards (5.0) being much stronger incentives for navigation.

### ✅ Fix #15: Episode Continuation After Reward Collection (CRITICAL for Learning)

**What was wrong:** Episodes ended immediately when reward was found, unlike IPPO/MAPPO which continue learning
**File:** `Environment.py` step() function  
**What changed:**

```python
# BEFORE (premature termination):
if self.reward_found:
    for agent in self.agents:
        truncations[agent] = True  # Episode ends immediately

# AFTER (continued learning like IPPO/MAPPO):
if not np.any(self.grid == 3):
    for agent in self.agents:
        truncations[agent] = True  # Episode ends when no rewards remain
```

**Impact:** Episodes now continue after reward collection, allowing all agents to learn from successful navigation patterns, matching IPPO/MAPPO behavior for better training stability.

---

## Architecture Now Correct

```
GridWorld Multi-Agent Environment (8 drones)
         ↓
JointAgentWrapper (concatenates obs, averages rewards)
         ↓
Single PPO Agent ("MlpPolicy")
         ↓
Normalized observations
         ↓
Centralized decision for all drones
```

## Expected Results

These fixes should produce:

- **15x15:** 85-95% success rate (was 56-70% with exploration but poor navigation)
- **30x30:** 60-80% success rate (was 20-40%)
- **45x45:** 30-50% success rate (was 5-15%)
- **Episode Lengths:** Start high (near max steps) and decrease as reward finding improves, matching IPPO/MAPPO behavior
- Stable training curves with strong reward-finding prioritization

## Files Modified

- ✅ `train_JOINT_PPO.py` (policy + vec_normalize save)
- ✅ `simulate_JOINT_PPO.py` (vec_normalize load + wrapper usage)
- ✅ `single_agent_wrapper.py` (observation consistency)
- ✅ `Environment.py` (reward shaping + episode continuation)
- ✅ `clear_joint_ppo_data.py` (training data cleanup script)

## Quick Commands

```bash
# Start training
python train_JOINT_PPO.py

# Run simulation
python simulate_JOINT_PPO.py

# Clear all training data (CAUTION!)
python clear_joint_ppo_data.py
```

## Testing Checklist

- [ ] Training creates: `joint_ppo_model_8drones.zip` + `joint_ppo_vec_normalize_8drones.pkl`
- [ ] No dimension mismatch errors during training
- [ ] Simulation metrics match training performance
- [ ] Agent count doesn't affect training behavior
