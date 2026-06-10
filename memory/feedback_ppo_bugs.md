# PPO Training Bugs — RedQueen

## Bug 1: lr_schedule not updated (CRITICAL) — FIXED 2026-05-29
**Symptom**: TensorBoard showed `learning_rate: 0.0003` for all 750k Stage 1 steps despite
intending 1.5e-4. Stage 1 avg_rank started at 2.22 (worse than random=1.5), never improved
past 1.75 in 750k steps.

**Root cause**: `_set_lr()` patched `model.learning_rate` and `optimizer.param_groups[lr]`
but NOT `model.lr_schedule`. SB3 calls `_update_learning_rate()` every rollout, which reads
`lr_schedule` (not `learning_rate`) to override the optimizer — silently restoring 3e-4.

**Fix**:
```python
def _set_lr(model, lr):
    model.learning_rate = lr
    model.lr_schedule = lambda _: lr          # <-- this line was missing
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = lr
```

## Bug 2: Episode continues after agent death (CRITICAL) — FIXED 2026-05-29
**Symptom**: ~80% of rollout transitions were useless "dead" transitions (action_mask forces
STOP, small negative rewards), diluting training data. With n_envs=4 and avg death at step
~100, 400+ wasted steps per episode.

**Root cause**: Engine returns `terminated = (alive_count <= 1)`. When our agent dies but
2+ enemies remain, `terminated=False` and the episode continues.

**Fix** in `src/wrappers/bomberland_env.py`:
```python
# Before:
terminated = engine_terminated
# After:
terminated = engine_terminated or (not our_alive)
```

## Bug 3: Bomb timer edge case — FIXED 2026-05-29
`timer==1` means the bomb **will explode next step** (not the current step). Danger
computation and reward attribution must treat `timer==1` as "imminent" — correct threshold
for `danger_now` channel and box-destruction attribution (box diff observed after timer hits 0).

## Bug 4: Blast expansion blocked by boxes — FIXED 2026-05-29
A box blocks **all cells beyond it** in the same blast direction. Blast expansion must stop
immediately upon hitting a box (box cell itself is affected; no cell further along that ray is).
Previous implementation incorrectly continued the ray past destroyed boxes.

## Bug 5: Self-play pool not updating — FIXED 2026-06-10
**Symptom**: Phase 4 self-play showed no improvement over continued curriculum; all opponents
were identical despite pool supposedly containing multiple snapshots.

**Root cause**: Snapshot filenames were hardcoded as `selfplay_best.pt` — each new save
overwrote the previous one. The sampling loop always loaded the same single file.

**Fix**: Timestamped filenames (`selfplay_{step}_{ts}.pt`), pool directory scanned at each
rollout to pick a random snapshot. Confirmed by logging loaded snapshot IDs in TensorBoard.

## Bug 6: Wrong ent_coef / optimizer state in self-play init — FIXED 2026-06-10
**Symptom**: Phase 4 started with ent_coef=0.08 (curriculum Stage 1 value) instead of 0.04
(final curriculum value), causing destructive entropy injection and policy regression.

**Root cause**: Self-play init code copied the Stage 1 ent_coef constant instead of reading
`STAGE_ENT_COEF[-1]` (the last stage value).

**Fix**: Phase 4 init now reads `ent_coef = STAGE_ENT_COEF[-1]` (= 0.04). Optimizer state
is also cleared at Phase 3->4 boundary to remove stale curriculum gradients.

## Confirmed Working Behavior (post all fixes — 2026-06-10)
- All 6 bugs fixed in ppo_trainer.py, bomberland_env.py, reward.py
- Stage 0 passed cleanly: avg_rank 0.26 <= 0.8 at 150k steps (rolling mean 0.255)
- Stage 1 converging properly with correct LR (1.5e-4) preserving BC weights

## Rule Added (2026-05-29)
**Rule 5**: After any edit to src/, check redqueen.ipynb to keep the notebook in sync.
