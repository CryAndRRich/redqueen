# /train — Check training status and launch training

Show latest checkpoints and training launch commands.

## Check status

```bash
ls -lt checkpoints/ | head -10       # latest checkpoints
ls -lt exports/ 2>/dev/null | head -5  # latest exports
```

## Launch Phase 3 — PPO Curriculum (7-stage)

```bash
python -m src.training.ppo_trainer --curriculum --init-from checkpoints/<bc_best>.pt
```

## Launch Phase 4 — Continuous Self-Play

```bash
python -m src.training.ppo_trainer --self-play --init-from checkpoints/ppo_curriculum_best.pt
```

## Note

On Kaggle, use `notebooks/train_on_kaggle.ipynb` instead of running these commands directly.
