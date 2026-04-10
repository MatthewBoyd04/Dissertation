# CRITICAL FIX - March 24, 2026: Episode Termination Bug

## Issue

Episodes were ending catastrophically early (~4-24 steps per episode) with 0% reward discovery on 30x30 and 45x45 maps.

## Root Cause

**File:** `single_agent_wrapper.py`, line in `step()` method

**Buggy Code:**

```python
joint_done = any(terminations.values()) or all(truncations.values())
```

**Problem:**
This line ends the **entire episode** as soon as **any single agent terminates** (hits a hazard).

With 8 drones and hazards scattered across the map:

1. Drone_1 hits a hazard at step 3 → terminations["Drone_1"] = True
2. `any(terminations.values())` → True
3. `joint_done` → True
4. Episode ends immediately, all remaining 7 drones are discarded
5. Result: 23 steps on 15x15, 4.9 steps on 30x30

## Correct Behavior

When an agent hits a hazard:

- That agent should be **removed from the environment**
- Other agents should **continue exploring**
- Episode should only end when:
  - ALL agents have been removed, OR
  - Maximum steps (truncation) reached for all agents

## Fix

**New Code:**

```python
# Episode ends ONLY when:
# 1. All agents have been removed/terminated (no observations returned), OR
# 2. All remaining agents are truncated (max steps reached)
# Individual agent terminations should NOT end the episode - only remove that agent
no_agents_left = len(obs_dict) == 0
all_truncated = all(truncations.values()) if truncations else False
joint_done = no_agents_left or all_truncated
```

## Expected Impact

With this fix, episodes should now run much longer:

- **15x15:** Up to 256 steps (maxCycles)
- **30x30:** Up to 1024 steps (maxCycles)
- **45x45:** Up to 2048 steps (maxCycles)

This allows the team of drones to:

- Continuously explore even when some drones fail
- Find the reward if it exists within the map bounds
- Achieve much better exploration coverage

## Why This Matters for Fair Comparison

IPPO and MAPPO likely don't have this bug because:

- IPPO trains each agent individually (policy per agent)
- MAPPO trains each agent with its own policy
- Joint PPO had this bug where episode termination was coupled to individual agents instead of to the environment state

This fix ensures Joint PPO is actually implemented correctly for fair algorithm comparison.
