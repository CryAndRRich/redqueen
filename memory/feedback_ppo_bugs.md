# PPO Training Bugs — RedQueen

## Bug 1: lr_schedule not updated (CRITICAL)
**Symptom**: TensorBoard showed `learning_rate: 0.0003` for all 750k Stage 1 steps despite
intending 1.5e-4. Stage 1 avg_rank started at 2.22 (worse than random=1.5), never improved
past 1.75 in 750k steps.

**Root cause**: `_set_lr()` patched `model.learning_rate` and `optimizer.param_groups[lr]`
but NOT `model.lr_schedule`. SB3 calls `_update_learning_rate()` every rollout, which reads
`lr_schedule` (not `learning_rate`) to override the optimizer — silently restoring 3e-4.

**FIXED 2026-05-29**:
```python
def _set_lr(model, lr):
    model.learning_rate = lr
    model.lr_schedule = lambda _: lr          # <-- this line was missing
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = lr
```

## Bug 2: Episode continues after agent death (CRITICAL)
**Symptom**: ~80% of rollout transitions were useless "dead" transitions (action_mask forces
STOP, small negative rewards), diluting training data. With n_envs=4 and avg death at step
~100, 400+ wasted steps per episode.

**Root cause**: Engine returns `terminated = (alive_count <= 1)`. When our agent dies but
2+ enemies remain, `terminated=False` and the episode continues.

**FIXED 2026-05-29** in `src/wrappers/bomberland_env.py`:
```python
# Before:
terminated = engine_terminated
# After:
terminated = engine_terminated or (not our_alive)
```

## Confirmed Working Behavior (post-fix)
- Stage 0 passed cleanly: avg_rank 0.26 <= 0.8 at 150k steps (rolling mean 0.255)
- Stage 1 expected to converge properly with correct LR (1.5e-4) preserving BC weights

## Bug 3: Bomb timer edge case (fixed)
**Note (2026-05-29)**: `timer==1` means the bomb **will explode next step** (not the current
step). Danger computation and reward attribution must treat `timer==1` as "imminent" — this is
the correct threshold for `danger_now` channel and for box-destruction attribution in the
reward function (box count diff observed in the step after timer hits 0).

## Bug 4: Blast expansion blocked by boxes (fixed)
**Note (2026-05-29)**: A box blocks **all cells beyond it** in the same blast direction.
The blast expansion must stop immediately upon hitting a box (the box cell itself is affected,
but no cell further along that ray is). Previous implementation incorrectly continued the ray
past destroyed boxes, granting false danger/attribution to cells behind them.

## New Rule Added (2026-05-29)
**Rule 5**: After any edit to src/, check redqueen.ipynb to keep the notebook in sync.
This prevents the notebook (used for Kaggle training runs) from diverging from fixed source.
