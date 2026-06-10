# Memory Index — RedQueen / GDGoC AI Challenge 2026

- [project_redqueen.md](project_redqueen.md) — Hybrid Ngoai Cong + Noi Nang architecture, 5-phase pipeline (TacticalAgent BC -> PPO Curriculum -> League Training), reward v4 (win=5.0, late-game multiplier, approach_enemy gated), ResNet CNN, VecNormalize, PFSP self-play, 5 Golden Rules
- [feedback_ppo_bugs.md](feedback_ppo_bugs.md) — 6 ppo_trainer bugs ALL FIXED (2026-06-10): lr_schedule not updated, episode runs after death, bomb timer edge case, blast blocked by boxes, self-play pool not updating, wrong ent_coef in self-play init
- [feedback_convergence.md](feedback_convergence.md) — 5 convergence lessons: self-play pool not updating, selfplay_best.pt overwriting destroys best policy, wrong ent_coef causes mode collapse, anchor agent replacement weakens curriculum, reward.py must use v4 sentinel values
