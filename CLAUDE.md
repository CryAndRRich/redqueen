# CLAUDE.md — RedQueen AI Agent System
> Auto-updated before each conversation compact via PreCompact hook.
> Last manual update: 2026-05-26

---

## 0. Quick Reference

```
ACTIONS      0=STOP  1=LEFT  2=RIGHT  3=UP  4=DOWN  5=PLACE_BOMB
MAP CODES    0=Grass  1=Wall  2=Box  3=ItemRadius  4=ItemCapacity
OBS SCHEMA   {"map": (13,13) int8, "players": (4,5) int8, "bombs": (N,4) int8}
PROTECTED    src/logic/action_masking.py  ← NEVER edit without explicit instruction

REWARD FILE  src/training/reward.py   (v4 logic, no versioned suffix)
NOTEBOOK     notebooks/train_on_kaggle.ipynb
CHECKPOINT   {phase}_{description}_{timestamp}.pt  ← never overwrite

COMMON COMMANDS:
  run match       python scripts/run_local_match.py
  export ONNX     python -m src.utils.export_onnx --checkpoint <ckpt>
  create zip      cd agent && zip -j ../submission.zip agent.py ../exports/model.onnx ../checkpoints/<ckpt>.pt
  validate zip    unzip -l submission.zip | grep agent.py   # must show agent.py NOT */agent.py
  tensorboard     tensorboard --logdir logs/
  tactical bc    python -m src.training.tactical_bc --generate --n-games 200
  tactical train python -m src.training.tactical_bc --train --epochs 20

KEY INVARIANTS:
  - agent.py must be at ZIP ROOT (not in a subfolder)
  - Never import PyTorch in agent.py — ONNX Runtime only
  - Bomb pad to (MAX_BOMBS=16, 4) before ONNX export
  - STAGE_ENT_COEF = [0.08, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]  ← Stage 1=0.10
  - STAGE_LR       = [3e-4, 1.5e-4, 1.2e-4, 1e-4, 8e-5, 8e-5, 5e-5]
  - MIN_STEPS_PER_STAGE = 100_000
  - Stage 0→1 transition: reload BC (not Stage 0 best) + clear optimizer state

REWARD v4 (canonical, as of 2026-06-10):
  win=5.0  death=-3.0  kill=2.5  box=0.5  late-game multiplier: smooth ramp ×1.0→×1.3 (steps 350→500)

WRAPPERS / NORMALIZATION:
  - VecNormalize applied to rewards at Phase 3+ (reward_normalization=True, norm_obs=False)
  - Prevents reward scale mismatch across curriculum stages

POLICY NETWORK (BomberCNNExtractor v2, as of 2026-06-10):
  - ResNet-style CNN blocks in spatial encoder (skip connections)
  - Orthogonal weight initialization throughout
  - Larger MLP heads: 256 hidden units (was 128)

SELF-PLAY OPPONENT SAMPLING (Phase 4, as of 2026-06-10):
  - PFSP (Prioritized Fictitious Self-Play): blends recency decay (α=0.9^age) with
    loss-rate weighting — opponents the agent loses to most are sampled more often
  - selfplay_best.pt tracks best avg_rank (not latest snapshot)
```

---

## 1. Project Context

**Competition**: GDGoC-HCMUS AI Challenge 2026
**Game**: Bomberland — 4-player FFA on a 13×13 grid, 6 discrete actions, 500 steps max
**Goal**: Build the strongest Bomberland agent via a 5-phase training pipeline culminating in League Training

**Tech stack** (non-negotiable):
- Python 3.11
- PyTorch — model definition and training only
- NumPy — all grid/observation processing; vectorized operations required
- ONNX + onnxruntime — production inference (must hit < 100 ms/step)
- Gymnasium — environment wrapper
- stable-baselines3 — PPO backbone

---

## 2. Competition Live Rules *(Last updated: 2026-05-26)*

### Step Processing Order (per step)

```
[Collect actions] → [Process movement] → [Place bombs] → [Decrease bomb timer]
→ [Resolve explosion] → [Remove dead agents] → [Spawn items] → [Check termination]
```

### Agent Actions

| Value | Action | Description |
|-------|--------|-------------|
| 0 | STOP | Stay in place |
| 1 | LEFT | Move left 1 cell |
| 2 | RIGHT | Move right 1 cell |
| 3 | UP | Move up 1 cell |
| 4 | DOWN | Move down 1 cell |
| 5 | PLACE_BOMB | Place bomb at current position |

**Movement rules:**
- Cannot enter Wall (code 1) or Box (code 2)
- Cannot enter a cell with an **active bomb from previous steps**. Exception: if agent just placed a bomb at that cell this step, they can still move out
- Multiple agents CAN occupy the same cell
- If ≥2 agents simultaneously step onto an item cell, the item is **destroyed** — no agent collects it

### Bomb Mechanics (CRITICAL)

**Placement condition**: `bombs_left > 0` AND current cell has no active bomb from a previous step.

**Duplicate bomb rule** (multiple agents place bomb on same cell in same step):
1. Priority goes to bomb with **larger radius**
2. Tie on radius → priority to agent with **smaller ID**
3. Only the "winning" agent has `bombs_left` decremented

**Explosion mechanics:**
- Spreads in 4 directions (cross pattern), blocked by Walls
- Boxes block and are destroyed by explosion
- Agents do NOT block explosions
- **Chain reaction**: explosion hitting another bomb triggers that bomb immediately in the same step

### Items

**From box destruction**: 30% Radius item, 30% Capacity item, 40% nothing

**Auto-spawn** (each empty cell, each step):
```
P = 0.0003 × (step / 165)
```
50% Radius, 50% Capacity.

**Simultaneous collection**: if ≥2 agents step on same item, item is destroyed, no one gets it.

### Elimination & End Conditions

- Agent is eliminated immediately if standing in a blast zone (including own bomb)
- Eliminated agents leave their bombs on the field
- Game ends when: (1) ≤1 agent survives, OR (2) step 500 reached

### Tie-break at Step 500

**BTC clarification (2026-05-26)**: When multiple agents survive to step 500, survivors are ranked **among themselves** (not all assigned equal rank). Same tie-break criteria apply:

```
1. Kills              (highest priority)
2. Boxes Destroyed
3. Items Collected
4. Bombs Placed       (lowest priority)
```

**Strategic consequence**: passive survival is not optimal. An agent with 1 kill and 0 boxes beats an agent with 0 kills and 10 boxes destroyed.

**evaluate() rank assignment at step 500**: `rank = 0` if `kills > 0`, else `rank = (n_alive - 1) / 2.0` (fractional rank for killed-nobody survivors).

### Submission Format (CRITICAL)

```
submission.zip
├── agent.py     ← MUST be at root, not inside any subfolder
├── model.onnx   ← primary inference
└── model.pt     ← TorchScript fallback (if onnxruntime unavailable)
```

**Do NOT include `requirements.txt`** — the evaluator rejects it (`requirements_txt_forbidden` error).

If `agent.py` is inside a subfolder (`submission/agent.py`), the evaluation engine cannot find it and the team scores ~0.

### Competition Environment Libraries (BTC official requirements.txt)

```
numpy, pygame, torch, tqdm, matplotlib, trueskill
google-auth>=2.0.0, google-auth-httplib2>=0.1.0, google-auth-oauthlib>=0.4.0
google-api-python-client>=2.0.0, pytest>=6.0.0, Flask>=2.2.0
tensorflow>=2.15.0, stable-baselines3>=2.2.1, gymnasium>=0.29.1
tensorboard>=2.15.0, scipy>=1.11.0, onnxruntime>=1.16.0
```

`onnxruntime` IS available. `sb3-contrib` is NOT listed — do not rely on it at inference time.

---

## 3. Directory Structure

```
redqueen/
├── agent/                        # All agent code
│   ├── agent.py                  # ★ Submission entry point — ONNX inference, self-contained
│   ├── genius_rule_agent.py      # Strongest baseline — teacher for BC + curriculum opponent
│   ├── tactical_rule_agent.py    # Advanced rule agent — curriculum stage 4
│   ├── trapper_rule_agent.py     # Trapper rule agent — curriculum stage 5
│   ├── smarter_rule_agent.py     # Medium rule agent — curriculum stage 3
│   ├── simple_rule_agent.py      # Simple rule agent — curriculum stage 1
│   ├── box_farmer_agent.py       # Box-focused agent — local testing opponent
│   ├── random_agent.py           # Random agent — curriculum stage 0
│   └── __init__.py               # Exports all baseline agents
│
├── src/                          # Core AI development
│   ├── logic/
│   │   ├── action_masking.py     # *** PROTECTED — see Golden Rules ***
│   │   └── pathfinding.py        # BFS / A* (vectorized NumPy)
│   ├── models/
│   │   └── policy_network.py     # BomberPolicyNet (actor-critic) + BomberCNNExtractor (SB3)
│   ├── training/
│   │   ├── tactical_bc.py        # Phase 0+2: TacticalAgent self-rollout + BC training
│   │   ├── ppo_trainer.py        # Phase 3-4: MaskablePPO + curriculum + self-play
│   │   └── reward.py             # Reward function (v4 logic, canonical name)
│   ├── wrappers/
│   │   └── bomberland_env.py     # Gymnasium single-agent wrapper for MaskablePPO
│   └── utils/
│       ├── feature_extractor.py  # obs dict → (spatial 15×13×13, aux 7) float32
│       └── export_onnx.py        # Export BomberPolicyNet → model.onnx
│
├── engine/                       # Game engine — READ-ONLY, never modify
│   └── game.py / map.py / bomb.py / player.py
│
├── scripts/                      # Helper scripts for local development & testing
│   ├── run_local_match.py        # Run headless or visual matches locally
│   ├── estimate_rankings.py      # Estimate TrueSkill rating vs random baselines
│   ├── replay_viewer.py          # Replay match JSON files (pygame + PIL)
│   ├── visualizer.py             # Live pygame viewer (called by run_local_match)
│   ├── agent_loader.py           # Dynamic agent.py loader (used by match scripts)
│   └── pre_compact_update.py     # Auto-updates CLAUDE.md before each compact
│
├── notebooks/
│   └── train_on_kaggle.ipynb     # Full training pipeline on Kaggle GPU
│
├── history_game/                 # 9,278 real competition match JSON files (gitignored subset)
│   └── YYYY-MM-DD/match_*.json   # See Section 5 for schema
│
├── checkpoints/                  # Timestamped .pt files (gitignored)
├── exports/                      # ONNX exports (gitignored)
├── data/                         # bc_dataset.npz (gitignored — can be 1GB+)
└── logs/                         # TensorBoard runs (gitignored)
```

---

## 4. Observation Schema (from `engine/game.py`)

```python
obs = {
    "map":     np.ndarray  # shape (13, 13), dtype int8
                           # 0=Grass, 1=Wall, 2=Box, 3=ItemRadius, 4=ItemCapacity
    "players": np.ndarray  # shape (4, 5), dtype int8
                           # columns: [x, y, alive, bombs_left, bomb_radius_bonus]
    "bombs":   np.ndarray  # shape (N, 4), dtype int8
                           # columns: [x, y, timer, owner_id]
}
```

**Engine constants** (from `engine/bomb.py`, `engine/player.py`):
- Bomb timer: 7 steps until explosion
- Max bomb radius bonus: 4 (radius = 1 + bonus, max radius = 5)
- Max bomb capacity: 5
- Starting positions: (1,1), (11,11), (1,11), (11,1) — corners, 1-indexed
- Movement: players CAN overlap each other, cannot enter Wall/Box/active Bomb tile

Action space: `{0: STOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: PLACE_BOMB}`

---

## 5. History Game Data

**Location**: `history_game/YYYY-MM-DD/match_*.json`
**Total files**: ~9,278 matches (as of 2026-05-23)
**Use**: Primary BC training data source. Do NOT run self-rollouts from scratch — use this first.

### Match File Schema

```python
{
  "seed": int,
  "team_ids": [str, str, str, str],      # UUID or "baseline_*" for baselines
  "meta": {"agent_names": [...]},
  "ranks": [int, int, int, int],          # rank[i] = placement of player i (0 = best)
  "survival_steps": [int, int, int, int], # steps survived by each player
  "runtime_stats": {                      # per-player: timeouts, errors, invalid_actions
    "0": {"timeouts": 0, "errors": 0, "invalid_actions": 0, "fallback_uses": 0},
    ...
  },
  "history": [                            # list of 501 frames (step 0 → 500)
    {
      "step": int,
      "actions": [int, int, int, int],    # None for step 0; action taken THIS step
      "alive": [bool, bool, bool, bool],
      "map": [[int, ...], ...],           # 13×13
      "players": [[x,y,alive,bombs_left,radius_bonus], ...],
      "bombs": [[x,y,timer,owner_id], ...]
    },
    ...
  ]
}
```

### Frame alignment (CRITICAL for BC)
- `history[t]["map/players/bombs"]` = **state AFTER** actions of step t were applied
- To get training pair: `obs_t = history[t]`, `action_t = history[t+1]["actions"][agent_id]`
- Correct: `obs = history[t]` → `label = history[t+1]["actions"][i]`

### BC Data Quality Filter
```python
# Keep only high-quality demonstrations
def is_quality_demo(match, agent_idx):
    return (
        match["ranks"][agent_idx] == 0          # winner
        and match["survival_steps"][agent_idx] >= 120  # survived long enough
        and sum(a == 5 for step in match["history"]
                for a in ([step["actions"][agent_idx]]
                          if step["actions"] else []))  >= 5  # placed bombs
    )
```

### Competitive Intelligence
- `baseline_*` agents: **0 wins** in 200 sampled games — usable as opponents, not as BC teachers
- Top teams by game count: `f9f492f0`, `cd455db7` — extract their winning demonstrations
- Average survival across all agents: ~175 steps (many early deaths from random agents)

---

## 6. Feature Engineering Specification (v2 — 4-player)

**Shape**: `(15, 13, 13)` spatial channels + `(7,)` auxiliary scalars

### Spatial Channels (15 total)

```
Ch 0   grass_mask          = (grid == 0).astype(float32)
Ch 1   wall_mask           = (grid == 1)
Ch 2   box_mask            = (grid == 2)
Ch 3   item_radius_mask    = (grid == 3)
Ch 4   item_capacity_mask  = (grid == 4)
Ch 5   my_position         = one-hot at (my_x, my_y) if alive
Ch 6   enemy1_position     = nearest alive enemy (by Manhattan distance)
Ch 7   enemy2_position     = second nearest alive enemy
Ch 8   enemy3_position     = third alive enemy (or zeros if fewer than 3 enemies)
Ch 9   bomb_timer          = timer/7.0 for each bomb cell (max if overlap)
Ch 10  bomb_owned          = 1.0 if bomb owned by me, else 0.0
Ch 11  bomb_enemy          = 1.0 if bomb owned by any enemy
Ch 12  danger_now          = blast tiles of bombs with timer <= 1
Ch 13  danger_soon         = blast tiles of bombs with timer <= 3 (includes danger_now)
Ch 14  danger_medium       = blast tiles of ALL active bombs
```

**Implementation constraint**: channels 12–14 must be computed via vectorized blast expansion — no Python loops over bombs. See `src/utils/feature_extractor.py`.

### Auxiliary Scalars (7 total)

```
0   my_bombs_left      / MAX_BOMB_CAPACITY (5)
1   my_radius_bonus    / (MAX_BOMB_RADIUS - 1) (4)
2   enemies_alive      / 3.0
3   step_normalized    = current_step / 500.0
4   boxes_remaining    = boxes_count / initial_boxes_count
5   my_kills_score     = kills / 3.0              # tracked externally, default 0
6   my_boxes_score     = boxes_destroyed / 20.0   # tracked externally, default 0
```

Scalars 5–6 require external stat tracking (not in raw obs). Use `0.0` during BC training, enable during PPO/League training when stats can be accumulated.

---

## 7. Reward Function (v3 — Tie-break + Game-mechanics Aware)

**File**: `src/training/reward.py` (v4 logic, canonical name).

### Reward Table

```python
REWARDS = {
    # === TERMINAL ===
    "win":                  3.0,    # last agent standing (also triggered on last kill)
    "agent_death":         -2.0,    # own death (immediate return)

    # === TIE-BREAK STATS (priority order: kills > boxes > items > bombs) ===
    "kill_credit":          2.0,    # confirmed kill; last-kill bonus: +win if enemies=0
    "box_destroyed":        0.4,    # per box destroyed by MY bomb (correctly attributed)
    "item_collected":       0.3,    # picking up radius or capacity item
    "bomb_placed":          0.003,  # placing a bomb (tie-break #4 padding)

    # === TACTICAL BONUSES ===
    "chain_reaction":       0.3,    # placing bomb adjacent to another active bomb (timer≤4)
    "item_contest":         0.1,    # moving toward item that an enemy is also approaching

    # === DANGER SHAPING ===
    "danger_evasion":       0.12,   # leaving blast zone (urgency ×1.5 if timer ≤ 3)
    "danger_enter":        -0.08,   # stepping INTO blast zone voluntarily
    "own_blast_loiter":    -0.04,   # standing in own bomb blast (× (8-timer) urgency)

    # === MOVEMENT ===
    "survival_step":        0.005,  # alive bonus per step
    "standing_still":      -0.008,  # repeated STOP actions
    "time_penalty":        -0.003,  # small time cost to prevent infinite games

    # === ENEMY PRESSURE (FFA-safe — do not raise) ===
    "approach_enemy":       0.006,  # × (prev_dist - curr_dist)
}
```

### Key changes from v2 → v3

| Change | v2 | v3 | Reason |
|--------|----|----|--------|
| `kill_credit` | 1.5 | **2.0** | Last kill grants win bonus (enemies=0 → +3.0) |
| `item_collected` | 0.15 | **0.3** | Tie-break #3 was underweighted |
| `chain_reaction` | — | **+0.3** | Chain kills = boxes + multi-kill synergy |
| `item_contest` | — | **+0.1** | Simultaneous collection destroys item |
| Box attribution | all boxes | **my bombs only** | Enemy bomb boxes gave false credit |

### Box destruction attribution (CRITICAL — fixed in v3)
```python
# Only credit boxes destroyed by MY bombs with timer==1 in prev_obs.
# timer==1 → decrements to 0 → explodes → boxes gone in curr_obs.
for row in prev_bombs:
    bx, by, timer, owner = ...
    if owner != agent_id or timer != 1:
        continue
    blast = _blast_tiles(grid, bx, by, radius)
    for tx, ty in blast:
        if prev_grid[tx, ty] == BOX and curr_grid[tx, ty] != BOX:
            my_boxes_destroyed += 1
```

---

## 8. Training Pipeline (v3 — 7-Stage Curriculum)

```
Phase 0  BC Data Generation   Generate BC data via TacticalAgent self-rollout
  └─ Script: python -m src.training.tactical_bc --generate --n-games 200
  └─ Source: TacticalRuleAgent self-play games
  └─ Estimated yield: 40,000–80,000 quality (obs, action) pairs

Phase 2  Behavioral Cloning Supervised from Phase 0 data
  └─ Script: python -m src.training.tactical_bc --train --epochs 20
  └─ Feature: 15-channel spatial + 7 aux scalars
  └─ Action Masking ACTIVE from this phase onward
  └─ Loss: Focal loss (γ=2) to handle class imbalance (PLACE_BOMB ~15%)
  └─ Early stopping: patience=5 epochs, no val_loss improvement
  └─ Checkpoint: bc_{epoch}ep_{timestamp}.pt

Phase 3  PPO + 7-Stage Curriculum   RL fine-tune from BC init
  └─ Action Masking: ACTIVE, hard-coded
  └─ Reward: v4 (win=5.0, death=-3.0, kill=2.5, box=0.5, smooth late-game multiplier)
  └─ VecNormalize: reward normalization active (norm_obs=False); prevents reward scale drift
     across curriculum stages
  └─ SubprocVecEnv: 4–8 parallel environments
  └─ ent_coef schedule: 0.08 → 0.10 → 0.08 → 0.07 → 0.06 → 0.05 → 0.04
     (Stage 1 = 0.10: hardest distribution shift Random→Simple; 0.08 caused entropy
      collapse -0.77→-0.41 within 3 iterations of Stage 1 start)
  └─ LR schedule (STAGE_LR): 3e-4 → 1.5e-4 → 1.2e-4 → 1e-4 → 8e-5 → 8e-5 → 5e-5
     (Lower LR at each stage: policy already near good solution, prevent drift;
      optimizer state cleared at stage start to remove Stage-N Adam momentum)
  └─ Stage 0→1 special: reload BC (not Stage 0 best) as Stage 1 init.
     Stage 0 specializes for Random opponents (weakens danger-avoidance vs strategic agents).
     BC has full GeniusRuleAgent knowledge → better Stage 1 foundation.
     (Advancement: rolling mean of last 3 eval windows ≤ threshold, min 100k steps)
  └─ 20% random opponent mixing per training env (prevents co-adaptation)
  └─ Curriculum stages (MIN_STEPS_PER_STAGE=100_000 each, eval every 50k over 200 games):
       Stage 0: random          avg_rank ≤ 0.8
       Stage 1: simple          avg_rank ≤ 1.2
       Stage 2: simple_smarter1 avg_rank ≤ 1.5  ← BRIDGE (1 Smarter + 2 Simple)
       Stage 3: smarter         avg_rank ≤ 1.8  (2 Smarter + 1 Simple anchor)
       Stage 4: tactical        avg_rank ≤ 2.0  (2 Tactical + 1 Smarter anchor)
       Stage 5: trapper         avg_rank ≤ 2.0  (2 Trapper + 1 Tactical anchor)
       Stage 6: genius          avg_rank ≤ 2.0  (2 Genius + 1 Tactical anchor)
  └─ Advance when: rolling mean of last 3 eval windows ≤ threshold AND min 100k steps
  └─ Checkpoint: ppo_{step}_{timestamp}.pt

Phase 4  Continuous Self-Play  PPO vs rolling Past Agents pool
  └─ Snapshot every 50k steps → Past Agents pool
  └─ PFSP opponent sampling: blends recency decay (α=0.9^age) with loss-rate weighting
     (opponents the current policy loses to most are upsampled — prevents stagnation)
  └─ Pool size: configurable via --pool-size CLI flag (default 20, forwarded correctly)
  └─ selfplay_best.pt tracks checkpoint with best avg_rank (not latest)
  └─ ent_coef=0.04, LR=5e-5 fixed at Phase 4 start (independent of PPO_DEFAULTS)
  └─ Opponents resample on each episode reset (not once per rollout batch)
  └─ VecNormalize reward normalization active (norm_obs=False, reward_normalization=True)
  └─ Checkpoint: selfplay_{step}_{timestamp}.pt

Phase 5  League Training   Full league with 4-player matchmaking
  └─ Game composition: [Candidate] + [Main] + [Past_i] + [Past_j]
  └─ The Gauntlet: 200 games, win-rate ≥ 55% for promotion
  └─ ONNX export before each Gauntlet run
  └─ Checkpoint: league_{step}_{timestamp}.pt
```

### Hyperparameter Baseline (PPO)

```python
PPO_DEFAULTS = {
    "learning_rate":   3e-4,
    "n_steps":         2048,     # per env per rollout
    "batch_size":      256,
    "n_epochs":        10,
    "gamma":           0.995,    # high gamma — bomb timer = 7 steps delay
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.03,     # base; overridden by STAGE_ENT_COEF at runtime
    "vf_coef":         0.5,
    "max_grad_norm":   0.5,
}

# Per-stage ent_coef (applied at stage start):
STAGE_ENT_COEF = [0.08, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
# Stage 1 = 0.10: Random→Simple is hardest distribution shift. 0.08 caused entropy
# collapse from -0.77 (Stage 0) to -0.41 within 3 iterations of Stage 1.
# Source: Costa 2021 "32 Details of PPO"; Meishner et al. 2019 (arXiv:1911.04947).

# Per-stage learning rate (optimizer state cleared at each stage transition):
STAGE_LR = [3e-4, 1.5e-4, 1.2e-4, 1e-4, 8e-5, 8e-5, 5e-5]
# Lower LR at each stage prevents distribution shift from destabilizing policy.
# Adam momentum cleared each transition — stale Stage-N gradients must not
# contaminate Stage-N+1 updates.
```

### Pipeline Summary Table

| Phase | Script | Key Flag | Checkpoint |
|-------|--------|----------|-----------|
| 0 | `tactical_bc.py` | `--generate` | `data/bc_dataset/` |
| 2 | `tactical_bc.py` | `--train` | `bc_{epoch}ep_{ts}.pt` |
| 3 | `ppo_trainer.py` | `--curriculum` | `ppo_{step}_{ts}.pt` |
| 4 | `ppo_trainer.py` | `--self-play` | `selfplay_{step}_{ts}.pt` |
| 5 | `league_trainer.py` | `--league` | `league_{step}_{ts}.pt` |

### Why the Bridge Stage Was Added (Stage 2: simple_smarter1)

Observed training failure (old 6-stage curriculum):
- Stage 1 (simple): avg_rank improved to 1.15 → advanced
- Stage 2 (smarter, old): avg_rank immediately collapsed to 2.46 → 2.58

Root cause: The jump from 3×SimpleRuleAgent to 2×SmarterRuleAgent+1×Simple was too large.
The bridge stage (1 Smarter + 2 Simple, threshold=1.5) smooths this transition.
Source: Territory Paint Wars (arXiv:2604.04983) — large opponent jumps cause policy collapse.

---

## 9. Strategic Priority Order

```
Priority 1: SURVIVE  (can't score if dead)
Priority 2: GET KILLS  (tie-break #1 — 1 kill beats 0 kills unconditionally)
Priority 3: DESTROY BOXES  (tie-break #2 — most likely differentiator in competitive play)
Priority 4: COLLECT ITEMS  (enables Priority 1, 2, 3 more effectively)
Priority 5: PLACE BOMBS  (tie-break #4 fallback)
```

**FFA-specific rules**:
- Do NOT approach an enemy fight if you are not the aggressor — let enemies kill each other
- If 2 enemies are in a fight, farm boxes elsewhere and claim the kill only if safe
- Late game (step > 350): bias heavily toward kills and boxes; survival is assumed

---

## 10. Strict Terminology

| Term | Definition |
|------|------------|
| **Action Masking** | Hard-logic binary filter that removes physically invalid or suicidal actions before the neural policy samples. Lives in `src/logic/action_masking.py`. |
| **Behavioral Cloning (BC)** | Supervised imitation learning (Phase 2) where the policy network is trained via focal cross-entropy loss to mimic demonstrations from `history_game/` and `GeniusRuleAgent`. |
| **Proximal Policy Optimization (PPO)** | On-policy RL algorithm (Phase 3–4) used to fine-tune the BC-initialised policy. Always combined with Action Masking. |
| **League Training** | Phase-5 training regime with three agent classes competing in a managed pool to produce increasingly robust policies. |
| **Continuous Self-Play** | Phase-4 training where the agent trains against a rolling pool of its own past checkpoints. |
| **Main Agent** | The current league champion. Serves as the active submission. |
| **Candidate Agent** | A new model currently in training. Must clear The Gauntlet to be promoted to Main. |
| **Past Agents** | Immutable frozen snapshots. Prevent mode collapse during self-play. |
| **The Gauntlet** | 200-game offline evaluation vs. Main + top-3 Past Agents. Win-rate ≥ 55% required. |
| **Nội Năng** | Internal Power — the neural network (policy + value networks). |
| **Ngoại Công** | External Power — hard rule-based layer (Action Masking, BFS Pathfinding, Danger Detection). |
| **Tie-break Stats** | The 4 criteria used when multiple players survive to step 500: Kills, Boxes Destroyed, Items Collected, Bombs Placed. |

---

## 11. Golden Rules

### Rule 1 — Action Masking is Sacred
> **Never modify `src/logic/action_masking.py` without an explicit instruction to do so.**

Any proposed change must be explicitly requested, then regression-tested against all edge cases before merging.

### Rule 2 — Checkpoint Immutability
> **Never overwrite an existing checkpoint. Always save with timestamped filename.**

Format: `{phase}_{description}_{timestamp}.pt`

### Rule 3 — The Gauntlet Before Promotion
> **No Candidate Agent becomes the Main Agent without passing The Gauntlet.**

Steps: 200 games → win-rate ≥ 55% → log to `logs/gauntlet/` with timestamp + model hashes.

### Rule 4 — Submission Zip Format
> **agent.py must be at the root of submission.zip, never inside a subfolder.**

Validate before every submission: `unzip -l submission.zip | grep agent.py` must show `agent.py` not `*/agent.py`.

### Rule 5 — Notebook Sync Required
> **After modifying any file in src/, review notebooks/train_on_kaggle.ipynb and update affected cells.**
> Every function call in the notebook must pass all parameters explicitly.

---

## 12. Coding Standards

### Type Hints and Docstrings — Required

```python
def compute_blast_mask(grid: np.ndarray, bx: int, by: int, radius: int) -> np.ndarray:
    """Return boolean mask of tiles within blast range, blocked by walls/boxes."""
    ...
```

### Vectorized NumPy — Required (no nested for-loops over grid)

```python
# FORBIDDEN:
for x in range(13):
    for y in range(13):
        if grid[x, y] == 2: danger[x, y] = True

# REQUIRED:
danger = (grid == 2)
```

For bomb blast expansion: use `np.zeros((H,W))` accumulation with `np.ix_` or explicit vectorized ray-marching, not Python iteration over grid cells.

### Inference Path (production)

```
obs dict → feature_extractor.py (NumPy only) → onnxruntime.InferenceSession → action
```

Never import PyTorch inside `agent.py`. Export to ONNX after every training phase.

### ONNX Compatibility

Bomb array has variable length — must pad to fixed shape `(MAX_BOMBS=16, 4)` with a validity mask before ONNX export. Handle this in `feature_extractor.py`.

### No Silent Fallbacks

Do not use `try/except` that returns a default action. If agent crashes, surface the traceback.

---

## 13. FAQ for AI Assistants

**Q: Can I refactor `action_masking.py` while fixing an unrelated bug?**
No. See Rule 1.

**Q: The model isn't converging — should I lower entropy coefficient?**
Diagnose first: check reward curves, verify action distribution isn't collapsed, confirm Action Masking isn't blocking all actions in edge states. Tune hyperparameters last.

**Q: Should I use PyTorch in `agent.py`?**
No. ONNX Runtime only. See Inference Path.

**Q: The tie-break rewards feel conflicting with survival rewards — which wins?**
Survival is still Priority 1 (you can't score tie-break points if dead). But past step ~350 with no kills, the reward bias should shift toward risky aggressive plays. This is handled by the `step_normalized` aux feature allowing the network to learn this implicitly.

**Q: How should I extract stats (kills, boxes) for reward computation?**
Track them externally: compare `alive` flags between obs frames for kill credit; compare `(grid == 2)` masks between frames for box destruction. The engine's `Player.stats` dict is not in the obs output.

**Q: Can I merge two checkpoints?**
No. League Training is the mechanism for combining knowledge across model versions.

---

## 14. Session Update Log
*(Auto-appended by `scripts/pre_compact_update.py` before each compact)*

### 2026-06-10 — Comprehensive bug fixes + modern RL improvements

**ppo_trainer.py fixes:**
- Fixed self-play pool not updating: opponents now resample on each episode reset (not once per rollout)
- Fixed `selfplay_best.pt` to track checkpoint with best avg_rank (was saving latest, not best)
- Fixed self-play phase using PPO_DEFAULTS ent_coef/LR — now correctly fixed at ent_coef=0.04, LR=5e-5
- Fixed `--pool-size` CLI flag not being forwarded into self-play loop (was silently ignored)
- Fixed 20% random opponent mixing to correctly preserve anchor agent in curriculum stages 4–6
- Fixed step counting to use `model.num_timesteps` (was double-counting in some paths)
- Added VecNormalize reward normalization (norm_obs=False, reward_normalization=True) at Phase 3+
- Added PFSP opponent sampling in self-play: blends recency decay with loss-rate weighting

**reward.py (v4 consolidated):**
- Canonical v4 values confirmed: win=5.0, death=-3.0, kill=2.5, box=0.5
- `approach_enemy` reward gated behind step > 300 (prevents early aggression before items collected)
- Late-game multiplier changed from hard step>400 cutoff to smooth ramp: ×1.0→×1.3 over steps 350→500
- Added bomb efficiency penalty: -0.05 for wasted bomb placement (no box/enemy in blast range)

**policy_network.py (BomberCNNExtractor v2):**
- Added ResNet-style skip connections to CNN spatial encoder blocks
- Applied orthogonal weight initialization throughout (policy, value, and CNN heads)
- Enlarged MLP heads from 128 to 256 hidden units (actor and critic independently)

**feature_extractor.py:**
- Verified and fixed blast expansion consistency: channels 12–14 now use identical ray-marching
  logic; previously ch 12/13 used a subtly different wall-blocking condition than ch 14

### 2026-05-29 — Codebase cleaned: dead code removed, canonical naming
- Renamed src/training/reward.py → src/training/reward.py (no version suffix)
- Deleted: src/training/bc_trainer.py (replaced by tactical_bc.py)
- Deleted: src/training/history_parser.py (no longer used)
- Fixed bomb timer edge case in reward.py (timer==1 = about to explode)
- Fixed blast expansion in feature_extractor.py (box blocks cells beyond it)
- All imports updated to use src.training.reward
- Notebooks cleaned: no hardcoded checkpoints, all params explicit

### 2026-05-29 — Full overhaul: TacticalAgent BC + Reward v4 + Training fixes
- Removed history_game BC; new tactical_bc.py for TacticalAgent self-rollout BC
- Reward v4: win=5.0, death=-3.0, kill=2.5, box=0.5, late-game 1.3x multiplier (step>400)
- Fixed: _set_lr() sets model.lr_schedule (lr no longer overridden per rollout)
- Fixed: stage transition loads best_stage_checkpoint (not latest)
- Fixed: optimizer state cleared on stage transition
- Added Rule 5: Notebook Sync Required

### 2026-05-24 — Initial architecture established
- Hybrid Nội Năng (PPO neural net) + Ngoại Công (Action Masking + BFS) architecture defined
- 5-phase pipeline: History Mining → BC → PPO+Curriculum → Self-Play → League Training
- Tie-break rule from BTC: Kills > Boxes > Items > Bombs
- history_game/ analyzed: 9,278 match files, avg survival 175 steps, baselines never win
- Reward v2 designed, feature engineering v2 (15 spatial + 7 aux), submission format clarified
- Legacy DQN encode_obs identified as 2-player only — rebuilt for 4-player

### 2026-05-26 — Training diagnostics + curriculum v3 + BC fix
- Entropy collapse fix: `ent_coef=0.03` insufficient → `STAGE_ENT_COEF = [0.08, 0.08, 0.07, 0.06, 0.06, 0.05, 0.04]`
- Bridge stage added: Stage 1→2 collapse (avg_rank 2.46→2.58) fixed by `simple_smarter1` (1 Smarter + 2 Simple, threshold=1.5); curriculum is now 7 stages
- Box attribution fixed in reward v3: only count boxes in blast zone of MY bombs with timer==1
- Reward v3: `kill_credit` 1.5→2.0, `item_collected` 0.15→0.3, added `chain_reaction` +0.3 and `item_contest` +0.1
- BC weight transfer fix: action head was being discarded during BC→PPO init (now correctly transferred)
- BTC tie-break clarification (2026-05-26): survivors at step 500 are ranked among themselves, not all equal. `evaluate()` updated: `rank=0` if `kills>0`, else `rank=(n_alive-1)/2.0`
- Notebook Cell 6 corrected: 100k min steps, rolling mean advancement, correct ent_coef schedule

### 2026-05-27 — Stage 1 convergence fix (3 root causes from Kaggle log)
- **Root cause 1**: LR=3e-4 too high at Stage 0→1 transition → entropy collapse -0.77→-0.41 in 3 iters
- **Root cause 2**: Stage 0 best weights specialized for Random agents → weak danger-avoidance vs Simple
- **Root cause 3**: Stale Adam momentum from Stage 0 contaminates Stage 1 gradients
- **Fix 1**: Added `STAGE_LR = [3e-4, 1.5e-4, 1.2e-4, 1e-4, 8e-5, 8e-5, 5e-5]` + `_set_lr()` helper
- **Fix 2**: `STAGE_ENT_COEF[1]` raised 0.08 → 0.10 (extra entropy insurance for hardest transition)
- **Fix 3**: Stage 0→1: reload BC weights (not Stage 0 best) + clear optimizer state each stage start
- Best Stage 1 result before fixes: avg_rank 1.30 at 650k, rolling mean never below 1.487 (threshold 1.2)

### 2026-05-27 — 2 critical bugs found from Kaggle run log + full code audit
- **Bug 1 (PRIMARY — LR never changed)**: `_set_lr()` patched `model.learning_rate` and `optimizer.param_groups[lr]` but NOT `model.lr_schedule`. SB3 calls `_update_learning_rate()` every rollout, which reads `lr_schedule` (not `learning_rate`) to override the optimizer — so intended 1.5e-4 Stage 1 LR was silently overwritten back to 3e-4 every rollout. TensorBoard confirmed: `learning_rate: 0.0003` for all 750k Stage 1 steps. Fix: add `model.lr_schedule = lambda _: lr` to `_set_lr()`.
- **Bug 2 (SECONDARY — dead agent floods buffer)**: Engine returns `terminated = (alive_count <= 1)`. When our agent dies but 2+ enemies remain, `terminated=False` and the episode continues for up to 400+ steps where the agent is dead (action_mask forces STOP, small negative rewards). With n_envs=4 and avg death at step ~100, ~80% of rollout transitions were useless "dead" transitions diluting training data. Fix: `bomberland_env.py.step()` now sets `terminated = engine_terminated or (not our_alive)`.
- **Evidence of Bug 1**: Stage 1 avg_rank at 50k = 2.22 (worse than random=1.5), never improved past 1.75 in 750k steps. With correct LR (1.5e-4), smaller initial KL → BC weights preserved → agent starts competent vs Simple agents.
- Stage 0 passed cleanly: avg_rank 0.26 ≤ 0.8 at 150k steps (rolling mean 0.255)

---

## 15. CLI Reference

### Running matches and evaluation

```bash
# Run a local headless match (default: RedQueen vs 3 random agents)
python scripts/run_local_match.py

# Run with specific agents and visual output
python scripts/run_local_match.py --agents agent/agent.py agent/genius_rule_agent.py --visual

# Estimate TrueSkill ranking against baselines
python scripts/estimate_rankings.py

# Replay a saved match JSON
python scripts/replay_viewer.py history_game/YYYY-MM-DD/match_0001.json
```

### Training pipeline

```bash
# Phase 0 — Generate BC data via TacticalAgent self-rollout
python -m src.training.tactical_bc --generate --n-games 200

# Phase 2 — Behavioral Cloning
python -m src.training.tactical_bc --train --epochs 20

# Phase 3 — PPO curriculum (init from BC checkpoint)
python -m src.training.ppo_trainer --curriculum --init-from checkpoints/bc_10ep_<ts>.pt

# Phase 4 — Continuous self-play (init from PPO checkpoint)
python -m src.training.ppo_trainer --self-play --init-from checkpoints/ppo_<step>_<ts>.pt
```

### Export and submission

```bash
# Export BomberPolicyNet to ONNX
python -m src.utils.export_onnx --checkpoint checkpoints/<ckpt>.pt --output exports/model.onnx

# Copy exports to agent/ then create zip (agent.py MUST be at zip root)
cp exports/model.onnx agent/model.onnx && cp exports/model.pt agent/model.pt
zip -j submissions/submission_$(date +%Y%m%d_%H%M).zip agent/agent.py agent/model.onnx agent/model.pt

# Validate zip — must show "agent.py" NOT "*/agent.py"
unzip -l submission.zip | grep agent.py
```

### Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir logs/
```
