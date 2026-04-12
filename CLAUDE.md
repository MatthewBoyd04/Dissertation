# CLAUDE.md — Dissertation: Multi-Agent Search-and-Rescue RL Simulation

## Token Efficiency Rules

- **Read before editing.** Never suggest or write changes to a file you haven't read in this session.
- **No speculative improvements.** Only change what is explicitly asked. No extra comments, docstrings, refactors, or "while I'm here" cleanup.
- **Use Grep/Glob for targeted lookups** — don't re-read whole files to find a symbol.
- **Use the Explore sub-agent** for open-ended codebase searches (e.g. "where is X used?", "what calls Y?") — this keeps raw search output out of the main context.
- **Use the Plan sub-agent** when designing non-trivial algorithmic changes (reward shaping, new observation channels, wrapper changes) before writing any code.
- **Don't repeat file contents back** in responses. Reference locations as `[file:line](path#Lline)`.
- **Keep responses concise.** Skip preamble and summaries unless asked.

## General Efficiency Rules

- **Utilise planning mode** If the user asks for a complex task to be completed Not in planning mode, suggest transitioning to planning mode first

## Sub-Agent Guidance

| Task                                                 | Use                          |
| ---------------------------------------------------- | ---------------------------- |
| Find where a function/class is used                  | Explore (quick)              |
| Understand MAPPO vs IPPO trade-offs for a change     | Plan                         |
| Broad architecture questions spanning multiple files | Explore (medium/thorough)    |
| Simple, targeted symbol lookup                       | Grep directly — no sub-agent |
| Known file read                                      | Read directly — no sub-agent |

---

## Project Overview

Dissertation project comparing **IPPO** (Independent PPO) vs **MAPPO** (Multi-Agent PPO) for a cooperative Search-and-Rescue (SAR) task. Drones explore a grid world to find reward tiles while avoiding hazards.

**Stack:** Python, Gymnasium, PettingZoo, Stable-Baselines3, PyTorch, Tkinter UI.

---

## Directory Structure

```
/                          ← repo root
├── IPPO/                  ← Independent PPO algorithm
│   ├── Environment.py     ← GridWorldEnvironment (PettingZoo ParallelEnv)
│   ├── Main_IPPO.py       ← Training loop entry point
│   ├── train_IPPO.py      ← trainAgents(), trainSingleAgent(), getMapChoice()
│   ├── simulate_IPPO.py   ← runSimulations() → CSV results
│   ├── single_agent_wrapper.py ← SingleAgentWrapper (gym.Env per drone)
│   ├── training_progress.json  ← Resumable training state
│   └── *_ppo_model.zip    ← Saved SB3 PPO models (one per drone)
├── MAPPO/                 ← Multi-Agent PPO algorithm
│   ├── Environment.py     ← Same GridWorldEnvironment (symlinked/copied)
│   ├── Main_MAPPO.py      ← Training loop entry point
│   ├── train_MAPPO.py     ← trainAgents(), MultiAgentWrapper, MAPPOWrapper
│   ├── simulate_MAPPO.py  ← runSimulations() → CSV results
│   ├── centralized_critic.py ← CentralizedCritic, MAPPOPolicy (CTDE)
│   ├── multi_agent_buffer.py ← Experience buffer for shared policy
│   ├── shared_mappo_model.zip ← Single shared PPO model for all drones
│   ├── vecnormalize.pkl   ← VecNormalize state (obs normalisation)
│   ├── training_progress.json
│   └── training_ui.py     ← (duplicate? — canonical UI is UI/training_ui.py)
├── UI/
│   └── training_ui.py     ← Tkinter GUI: launch training, tune reward weights, view plots
├── Maps.py                ← map_15x15, map_30x30, map_45x45 (reads training_config.json for safe/hazard mode)
├── PlotAnalysis.py        ← plotAnalysisData() — matplotlib plots from CSV
├── LoggerConfig.py        ← log = Log(LogLevel.INFO) via BetterDebugging
├── training_config.json   ← Shared runtime config (written by UI, read by training scripts)
├── requires.txt           ← pip dependencies
└── Archives/              ← Old data/experiments — EXCLUDE from all analysis
```

---

## Core Environment: `GridWorldEnvironment`

**File:** [IPPO/Environment.py](IPPO/Environment.py)

### Grid tile values

| Value | Meaning                                            |
| ----- | -------------------------------------------------- |
| 0     | Available (traversable)                            |
| 1     | Blocked (wall)                                     |
| 2     | Hazard (fatal, optional — disabled in "safe" mode) |
| 3     | Reward tile (target)                               |

### Key parameters

- `mapPreset`: 2D list (from `Maps.py`)
- `num_drones`: 1–8, clamped
- `maxCycles`: episode step limit (128/512/1024 by map size in IPPO)
- `visionRange`: default 2 → 5×5 observation window
- `use_map_memory`: adds 6th obs channel (agent's cumulative map memory, -1/0/1)
- `render_every`: render every N episodes (uses `graphics` library)

### Observation space

Shape `(C, 5, 5)` where C=5 (or 6 with map memory):

- Ch 0: terrain (blocked/hazard)
- Ch 1: globally discovered tiles
- Ch 2: self position (centre = 1.0)
- Ch 3: other agents' positions
- Ch 4: reward tiles
- Ch 5 (optional): agent's personal map memory

### Action space

`Discrete(5)`: 0=Up, 1=Right, 2=Down, 3=Left, 4=Stay

### Reward weights (defaults, overridable via `training_config.json`)

- `tileDiscovered`: 1 (per newly discovered tile)
- `rewardFound`: 100 (IPPO) / 10000 (MAPPO)
- `HazardHit`: -100
- `Steps`: -0.01

### MAPPO-only reward weights

`individualDiscovery`, `explorationBonus`, `spacingBonus`, `noveltyBonus`

### Starting positions (by drone index)

Left-centre, right-centre, top-centre, bottom-centre, top-right, bottom-right, top-left, bottom-left.

---

## IPPO Algorithm

**Entry:** `IPPO/Main_IPPO.py` → `train_IPPO.trainAgents()` → `simulate_IPPO.runSimulations()`

- Each drone gets its **own SB3 PPO model** (`Drone_N_ppo_model.zip`)
- `SingleAgentWrapper` converts the multi-agent env to single-agent gym.Env for SB3
- **Frozen agents:** during training of agent N, other agents optionally load their last-saved models (stabilises training)
- `use_frozen_agents` toggled by `training_config.json` (`"ippo_frozen"` vs `"ippo"`)
- Training is **sequential** per drone (parallel=False default) or threaded (parallel=True)
- Curriculum: `cirriculum_Random` — 15x15 for first 2.5M steps, then mix 15x15/30x30, then all three

---

## MAPPO Algorithm

**Entry:** `MAPPO/Main_MAPPO.py` → `train_MAPPO.trainAgents()` → `simulate_MAPPO.runSimulations()`

- **Single shared PPO model** for all drones (`shared_mappo_model.zip`)
- CTDE: decentralized actor + `CentralizedCritic` (sees all agents' obs concatenated)
- `MultiAgentWrapper`: rotates through agents each step, uses shared policy for all non-current agents
- `VecNormalize` on observations only (`norm_reward=False`) — state saved to `vecnormalize.pkl`
- 8 parallel envs (`DummyVecEnv`)
- Entropy decay callback: 0.1 → 0.01 over training
- Curriculum: `cirriculum_Random` — 15x15 for first 500K steps, then mixes 30x30, then 45x45

---

## Training Config (`training_config.json`)

Written by the Tkinter UI, read at import time by `Maps.py`, `Environment.__init__`, and `Main_*.py`:

```json
{
  "algorithm": "mappo",        // "ippo", "ippo_frozen", "mappo"
  "num_drones": 8,
  "map_mode": "safe",          // "safe" (no hazards) | "hazard"
  "reward_weights": { ... }
}
```

**Important:** `Maps.py` reads this at module import time — if the config changes, the module must be re-imported/reloaded for it to take effect.

---

## Analysis & Results

- `runSimulations()` runs N episodes and appends a row to `{map_name}_analysis_Results.csv`
- CSV columns: `TimeSteps, Reward Found %, Avg Steps, Avg Tiles, Avg Tiles Per Step, Avg Steps to Reward, Avg Score`
- `PlotAnalysis.plotAnalysisData(csv_path)` plots all metrics with moving averages (matplotlib, interactive mode)
- `getEpisodeAnalysis()` in Environment returns per-episode dict

---

## UI (`UI/training_ui.py`)

Tkinter app that:

1. Configures and writes `training_config.json`
2. Launches `Main_IPPO.py` or `Main_MAPPO.py` as a subprocess
3. Streams stdout/stderr to a scrolled text widget (ANSI colour parsing included)
4. Embeds live matplotlib plots of the CSV results
5. Can reset training (delete model + progress files)

---

## Key Dependencies (`requires.txt`)

```
gymnasium
BetterDebugging    ← custom logger (log.i/log.d/log.e)
matplotlib
pettingzoo
torch
stable_baselines3
```

Also used but not listed: `graphics` (for env rendering), `pandas`, `tkinter` (stdlib).

---

## Common Tasks & Where to Look

| Task                         | Files                                                                    |
| ---------------------------- | ------------------------------------------------------------------------ |
| Change reward shaping        | `Environment.py` reward weights / `training_config.json`                 |
| Add observation channel      | `Environment.__init__` obs space, `_get_obs()`, `use_map_memory` pattern |
| Change map layout            | `Maps.py`                                                                |
| Tune PPO hyperparams (IPPO)  | `train_IPPO.trainSingleAgent()` model creation block                     |
| Tune PPO hyperparams (MAPPO) | `train_MAPPO.trainAgents()` model creation block                         |
| Change curriculum schedule   | `getMapChoice()` in `train_IPPO.py` or `train_MAPPO.py`                  |
| Add new analysis metric      | `Environment.getEpisodeAnalysis()` + CSV writer in `simulate_*.py`       |
| UI config options            | `UI/training_ui.py` + `training_config.json` schema                      |
