# Convergence Lessons — RedQueen Self-Play & Curriculum

## Lesson 1: Self-play pool not updating = zero benefit from Phase 4
**Problem**: If the opponent pool snapshot is saved but the training loop does not reload
opponents from the pool each rollout (or does so incorrectly), all games are played vs the
same frozen opponent — equivalent to curriculum stage 6 repeated. The agent gets no
diversity signal, overfits to one playstyle, and self-play provides zero additional benefit
over continued curriculum training.

**Fix**: Verify opponent sampling loop actually loads a fresh random snapshot from the pool
directory each episode/rollout. Log which snapshot is loaded; confirm the set of snapshot
IDs rotates in TensorBoard.

---

## Lesson 2: selfplay_best.pt overwriting destroys best policy
**Problem**: Saving every new checkpoint as `selfplay_best.pt` (fixed filename) means a
later, worse checkpoint silently overwrites the actual best policy. After a regression, the
best weights are unrecoverable.

**Fix**: Always save with timestamped filenames: `selfplay_{step}_{timestamp}.pt`.
Track best by a separate metric log (e.g., CSV with step, avg_rank, elo). Never use a
fixed "best" filename in Phase 4 or Phase 5.

---

## Lesson 3: Wrong ent_coef in self-play causes mode collapse
**Problem**: Resetting ent_coef to a high value (e.g., 0.08) at the start of Phase 4
self-play after curriculum has tuned the policy causes destructive entropy injection.
Conversely, using the default (0.03) when the policy just finished curriculum at 0.04
introduces an unintended step-down that can destabilize early self-play.

**Fix**: Start Phase 4 with ent_coef = final curriculum stage value (0.04). Only decay
further if entropy is still too high after 200k self-play steps. Document the chosen value
in the checkpoint filename or a sidecar .json.

---

## Lesson 4: Anchor agent replacement in mixing weakens curriculum
**Problem**: The curriculum mixing rule (20% random opponent per env) is intended to prevent
co-adaptation. If the "anchor" slot (the weaker/random agent) is accidentally replaced with
a snapshot of the current policy, all 4 env slots play the same policy variant — the
diversity guarantee is broken and the agent can overfit to a symmetric opponent.

**Fix**: Hard-code the anchor slot to always load from `agent/random_agent.py` (or the
stage's designated anchor). Assert in the training loop that at least 1 of the 4 env
opponents differs from the current policy class.

---

## Lesson 5: reward.py must use v4 content (not old v3)
**Problem**: After the rename/rewrite cycle, `src/training/reward.py` was momentarily
reverted to v3 content (win=3.0, death=-2.0, no late-game multiplier) while the filename
and imports still pointed to the "v4" canonical path. Training ran with v3 rewards silently.

**Fix**: Canonical values to verify in `src/training/reward.py`:
```python
"win":          5.0    # NOT 3.0
"agent_death": -3.0    # NOT -2.0
"kill_credit":  2.5    # NOT 2.0
"box_destroyed": 0.5   # NOT 0.4
# Late-game multiplier active (step > 400 -> scale *= 1.3)
# approach_enemy gated: only reward if enemy is within actionable range
```
After any reward.py edit, grep for these sentinel values before starting a training run.
