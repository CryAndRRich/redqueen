"""
Phase 3–4 — PPO trainer using sb3-contrib MaskablePPO.

Phase 3 (--curriculum): train against rule-based opponents with curriculum
Phase 4 (--self-play):  train against rolling Past Agent snapshots

Requires: pip install sb3-contrib

Usage:
    # Phase 3 — curriculum PPO from BC init
    python -m src.training.ppo_trainer --curriculum --init-from checkpoints/bc_best.pt

    # Phase 4 — self-play
    python -m src.training.ppo_trainer --self-play --init-from checkpoints/ppo_curriculum_best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Callable

import os
import numpy as np

# Suppress TF/XLA/gRPC C++ warnings that fire when CUDA workers start.
# Must be set before any subprocess spawns so workers inherit these values.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─────────────────────────────────────────────────────────────────────────── #
# Imports (heavy; only at runtime)                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_maskable_ppo():
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        return MaskablePPO, MaskableEvalCallback
    except ImportError as e:
        raise ImportError(
            "sb3-contrib not found. Install with: pip install sb3-contrib"
        ) from e


# ─────────────────────────────────────────────────────────────────────────── #
# PPO hyperparameters                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

PPO_DEFAULTS: dict = {
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      256,
    "n_epochs":        10,
    "gamma":           0.995,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.03,   # raised from 0.02: BC pretraining makes policy deterministic;
                               # 0.02 causes entropy collapse within Stage 1 (entropy_loss -1.16→-0.58)
    "vf_coef":         0.5,
    "max_grad_norm":   0.5,
}

# Curriculum stages: (opponent_fn_name, avg_rank_threshold)
# avg_rank: 0=win, 1=2nd, 2=3rd, 3=died early — lower is better.
# Opponents include an anti-forgetting anchor (1 easier agent) from Stage 3 onward.
# Stage 2 is a bridge stage (1 Smarter + 2 Simple) that smooths the Simple→Smarter jump.
# Without the bridge, Stage 1→2 is too large and causes avg_rank collapse (observed: 2.46→2.58).
# Ref: IJCAI 2024 workshop — rule-based anchors prevent catastrophic forgetting.
# Ref: Territory Paint Wars (arXiv:2604.04983) — large opponent jumps cause policy collapse.
CURRICULUM_STAGES = [
    ("random",          0.8),   # Stage 0: must dominate randoms
    ("simple",          1.2),   # Stage 1: must clearly beat simple
    ("simple_smarter1", 1.5),   # Stage 2 (BRIDGE): 1 Smarter + 2 Simple — smooth transition
    ("smarter",         1.8),   # Stage 3: 2 Smarter + 1 Simple anchor
    ("tactical",        2.0),   # Stage 4: 2 Tactical + 1 Smarter anchor
    ("trapper",         2.0),   # Stage 5: 2 Trapper + 1 Tactical anchor
    ("genius",          2.0),   # Stage 6: 2 Genius + 1 Tactical anchor
]

# ent_coef per stage: start high (0.08) to undo BC determinism, anneal across stages.
# BC pretraining drives policy toward near-deterministic distribution (low entropy).
# ent_coef=0.03 was insufficient — entropy collapsed from -1.16→-0.30 within Stage 1.
# Ref: Costa 2021 "32 Details of PPO" — entropy bonus is primary anti-collapse mechanism.
# Ref: Meishner et al. 2019 (arXiv:1911.04947) — BC init requires elevated entropy pressure.
STAGE_ENT_COEF = [0.08, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03]

# Minimum steps per stage regardless of win-rate threshold being met.
# Prevents premature advancement before skills are consolidated.
# Ref: Meishner et al. 2019 — 200k steps minimum for consolidation.
MIN_STEPS_PER_STAGE = 200_000


# ─────────────────────────────────────────────────────────────────────────── #
# Opponent factories                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_opponents(stage_name: str, opp_ids: list[int], mix_random: bool = False):
    """
    Return list of 3 opponent agents for a given curriculum stage.

    mix_random: if True, replace one opponent with RandomAgent with 20% probability.
    This prevents co-adaptation with specific opponents (Territory Paint Wars, 2024).
    Only applied to training envs, not evaluation envs.
    """
    from agent import (
        GeniusRuleAgent, TacticalRuleAgent, TrapperRuleAgent,
        SmarterRuleAgent, SimpleRuleAgent, RandomAgent,
    )

    if stage_name == "random":
        return [RandomAgent(i) for i in opp_ids]
    if stage_name == "simple":
        opps = [SimpleRuleAgent(i) for i in opp_ids]
    elif stage_name == "simple_smarter1":
        # Bridge: 1 Smarter + 2 Simple — gradual introduction of stronger opponent
        opps = [SmarterRuleAgent(opp_ids[0]), SimpleRuleAgent(opp_ids[1]), SimpleRuleAgent(opp_ids[2])]
    elif stage_name == "smarter":
        # 2 smarter + 1 simple anchor — prevents forgetting Stage 1 survival skills
        opps = [SmarterRuleAgent(opp_ids[0]), SmarterRuleAgent(opp_ids[1]), SimpleRuleAgent(opp_ids[2])]
    elif stage_name == "tactical":
        # 2 tactical + 1 smarter anchor
        opps = [TacticalRuleAgent(opp_ids[0]), TacticalRuleAgent(opp_ids[1]), SmarterRuleAgent(opp_ids[2])]
    elif stage_name == "trapper":
        # 2 trapper + 1 tactical anchor
        opps = [TrapperRuleAgent(opp_ids[0]), TrapperRuleAgent(opp_ids[1]), TacticalRuleAgent(opp_ids[2])]
    elif stage_name == "genius":
        # 2 genius + 1 tactical anchor
        opps = [GeniusRuleAgent(opp_ids[0]), GeniusRuleAgent(opp_ids[1]), TacticalRuleAgent(opp_ids[2])]
    elif stage_name == "mixed_t_g":
        opps = [TacticalRuleAgent(opp_ids[0]), TacticalRuleAgent(opp_ids[1]), GeniusRuleAgent(opp_ids[2])]
    else:
        raise ValueError(f"Unknown stage: {stage_name}")

    # 20% random mixing: replace last opponent with RandomAgent.
    # Prevents policy from overfitting to specific opponent behaviors (co-adaptation).
    if mix_random and random.random() < 0.20:
        opps[-1] = RandomAgent(opp_ids[-1])

    return opps


# ─────────────────────────────────────────────────────────────────────────── #
# Environment factory                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _make_env_fn(stage_name: str, seed: int, agent_id: int = 0, mix_random: bool = False) -> Callable:
    """Return a callable that creates a SingleAgentBomberEnv (for VecEnv).
    mix_random=True for training envs only; always False for eval envs.
    """
    def _factory():
        from src.wrappers.bomberland_env import SingleAgentBomberEnv
        all_ids = list(range(4))
        all_ids.remove(agent_id)
        opps = _make_opponents(stage_name, all_ids, mix_random=mix_random)
        env = SingleAgentBomberEnv(opponents=opps, agent_id=agent_id, seed=seed)
        return env
    return _factory


def _make_self_play_env_fn(
    snapshot_dir: Path,
    seed: int,
    agent_id: int = 0,
) -> Callable:
    """Create env where 3 opponents are sampled from Past Agent snapshots."""
    def _factory():
        from src.wrappers.bomberland_env import SingleAgentBomberEnv
        from src.inference.past_agent import PastAgentWrapper  # see below

        all_ids = list(range(4))
        all_ids.remove(agent_id)

        snapshots = sorted(snapshot_dir.glob("*.pt"))
        if not snapshots:
            # Diverse fallback before any snapshots exist: mix tactical + genius
            from agent import GeniusRuleAgent, TacticalRuleAgent, TrapperRuleAgent
            opps = [TrapperRuleAgent(all_ids[0]), TacticalRuleAgent(all_ids[1]), GeniusRuleAgent(all_ids[2])]
        else:
            # Exponential-decay weighted sampling (recent snapshots preferred)
            weights = np.array([0.9 ** i for i in range(len(snapshots) - 1, -1, -1)])
            weights /= weights.sum()
            chosen = np.random.choice(snapshots, size=3, replace=True, p=weights)
            opps = [PastAgentWrapper(str(c), i) for c, i in zip(chosen, all_ids)]

        env = SingleAgentBomberEnv(opponents=opps, agent_id=agent_id, seed=seed)
        return env
    return _factory


# ─────────────────────────────────────────────────────────────────────────── #
# BC weight transfer into SB3 model                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _load_bc_into_sb3(model, bc_path: Path, device: str) -> None:
    """Transfer BC-trained weights into SB3 MaskablePPO policy."""
    import torch
    from src.models.policy_network import BomberPolicyNet

    bc_net = BomberPolicyNet.load(str(bc_path), device=device)
    fe = model.policy.features_extractor
    fe.spatial_enc.load_state_dict(bc_net.spatial_enc.state_dict())
    fe.aux_enc.load_state_dict(bc_net.aux_enc.state_dict())
    fe.fusion.load_state_dict(bc_net.fusion[:2].state_dict())  # first Linear+ReLU
    print(f"Loaded BC weights from {bc_path.name}")


# ─────────────────────────────────────────────────────────────────────────── #
# Past Agent wrapper (for self-play)                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _write_past_agent_wrapper():
    """Write src/inference/past_agent.py if not present."""
    dest = _ROOT / "src" / "inference" / "past_agent.py"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.parent.joinpath("__init__.py").touch()
    dest.write_text(
        '''"""Thin wrapper: load a BomberPolicyNet checkpoint and expose .act(obs) interface."""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from src.models.policy_network import BomberPolicyNet
from src.utils.feature_extractor import extract_features
from src.logic.action_masking import compute_action_mask, apply_mask_to_logits


class PastAgentWrapper:
    """Rule-compatible .act(obs) interface backed by a BomberPolicyNet checkpoint."""

    def __init__(self, ckpt_path: str, agent_id: int) -> None:
        self.agent_id = int(agent_id)
        self._net = BomberPolicyNet.load(ckpt_path, device="cpu")

    def act(self, obs: dict) -> int:
        spatial, aux = extract_features(obs, self.agent_id)
        mask = compute_action_mask(obs, self.agent_id)
        sp_t = torch.from_numpy(spatial).unsqueeze(0)
        ax_t = torch.from_numpy(aux).unsqueeze(0)
        with torch.no_grad():
            logits = self._net.get_action_logits(sp_t, ax_t).squeeze(0).numpy()
        masked = apply_mask_to_logits(logits, mask)
        return int(np.argmax(masked))
'''
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Win-rate evaluation callback                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class CurriculumAdvanceCallback:
    """
    Checks average rank after every learn() call.
    Rank: 0=win (sole survivor), 1=2nd, 2=3rd, 3=died early — lower is better.

    Advances stage when ALL of the following hold:
      1. avg_rank <= threshold for `patience` consecutive windows
      2. No worsening trend (rank delta > 0.15 over last 3 evals resets counter)
      3. At least `min_steps` have been taken in this stage (consolidation guard)
    """

    def __init__(
        self,
        eval_env_fn: Callable,
        n_eval_episodes: int = 200,   # raised from 100: 4-player FFA has high variance
        rank_threshold: float = 1.5,
        patience: int = 3,
        min_steps: int = MIN_STEPS_PER_STAGE,
    ) -> None:
        self.eval_env_fn = eval_env_fn
        self.n_eval_episodes = n_eval_episodes
        self.threshold = rank_threshold
        self.patience = patience
        self.min_steps = min_steps
        self._consecutive = 0
        self._history: list[float] = []
        self._steps_in_stage: int = 0
        self.stage_passed: bool = False

    def evaluate(self, model) -> float:
        """Run n_eval_episodes, return average rank (lower is better)."""
        env = self.eval_env_fn()
        total_rank = 0
        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            raw = env.raw_obs
            if raw is not None:
                alive = [int(raw["players"][i][2]) for i in range(4)]
                n_alive = sum(alive)
                # rank 0=win (sole survivor), 1=2nd place, ..., 3=died early
                rank = n_alive - 1 if alive[0] else 3
            else:
                rank = 3
            total_rank += rank
        env.close()
        return total_rank / self.n_eval_episodes

    def check(self, model, steps_this_iter: int = 50_000) -> bool:
        """Return True if curriculum should advance."""
        self._steps_in_stage += steps_this_iter
        avg_rank = self.evaluate(model)
        self._history.append(avg_rank)
        print(f"  Avg rank: {avg_rank:.2f} (threshold ≤ {self.threshold:.1f})")

        if avg_rank <= self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        # Trend guard: worsening by >0.15 over the last 3 evals resets counter.
        # Prevents advancing while the policy is deteriorating.
        if len(self._history) >= 3:
            trend = self._history[-1] - self._history[-3]
            if trend > 0.15 and self._consecutive > 0:
                print(f"  ↓ Worsening trend ({trend:+.2f} over 3 evals) — resetting consecutive count")
                self._consecutive = 0

        # Consolidation guard: enforce minimum steps before advancing
        if self._steps_in_stage < self.min_steps:
            print(f"  ⏳ Min steps not reached ({self._steps_in_stage:,}/{self.min_steps:,}), holding")
            return False

        self.stage_passed = self._consecutive >= self.patience
        return self.stage_passed


# ─────────────────────────────────────────────────────────────────────────── #
# Main training functions                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def train_curriculum(
    output_dir: Path,
    total_steps_per_stage: int = 500_000,
    n_envs: int = 8,
    init_from: Path | None = None,
    device: str = "auto",
) -> tuple[Path, bool]:
    """
    Phase 3: curriculum PPO training.

    Returns:
        (best_ckpt, all_stages_passed) — if a stage fails to meet threshold within
        its budget, training stops immediately at that stage.
        all_stages_passed=False means self-play should NOT be run.
    """
    import warnings
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv

    # Suppress SB3 net_arch deprecation warning
    warnings.filterwarnings("ignore", message=".*shared layers.*", category=UserWarning)

    MaskablePPO, _ = _make_maskable_ppo()
    _write_past_agent_wrapper()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models.policy_network import BomberCNNExtractor, make_observation_space

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "ppo_curriculum_best.pt"

    model = None
    all_stages_passed = True

    for stage_idx, (stage_name, wr_thresh) in enumerate(CURRICULUM_STAGES):
        print(f"\n=== Curriculum Stage {stage_idx}: {stage_name} (avg_rank threshold ≤ {wr_thresh:.1f}) ===")

        # Training envs use mix_random=True (20% random opponent substitution).
        # Eval env uses mix_random=False for consistent, deterministic evaluation.
        env_fns = [_make_env_fn(stage_name, seed=stage_idx * 1000 + i, mix_random=True) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)

        if model is None:
            model = MaskablePPO(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs={
                    "features_extractor_class": BomberCNNExtractor,
                    "features_extractor_kwargs": {"features_dim": 256},
                    "net_arch": dict(pi=[128], vf=[128]),
                },
                verbose=1,
                device=device,
                tensorboard_log=str(output_dir / "tb_logs"),
                **PPO_DEFAULTS,
            )
            if init_from and init_from.exists():
                _load_bc_into_sb3(model, init_from, device)
            # Apply stage-specific ent_coef from Stage 0 — PPO_DEFAULTS ent_coef is
            # too low to counteract BC-pretraining determinism (observed entropy collapse).
            model.ent_coef = STAGE_ENT_COEF[stage_idx]
            print(f"  ent_coef set to {STAGE_ENT_COEF[stage_idx]} for stage {stage_idx}")
        else:
            model.set_env(vec_env)
            # Re-inject exploration entropy when entering a harder stage.
            ent_coef = STAGE_ENT_COEF[stage_idx]
            model.ent_coef = ent_coef
            print(f"  ent_coef reset to {ent_coef} for stage {stage_idx}")

        eval_fn = _make_env_fn(stage_name, seed=9999, mix_random=False)
        cb = CurriculumAdvanceCallback(
            eval_fn,
            rank_threshold=wr_thresh,
            patience=3,
            min_steps=MIN_STEPS_PER_STAGE,
            # n_eval_episodes uses default (200)
        )

        steps_done = 0
        while steps_done < total_steps_per_stage:
            model.learn(
                total_timesteps=50_000,
                reset_num_timesteps=False,
                tb_log_name=f"stage_{stage_idx}_{stage_name}",
            )
            steps_done += 50_000

            ts = time.strftime("%Y%m%d_%H%M%S")
            ckpt = output_dir / f"ppo_s{stage_idx}_{steps_done}steps_{ts}.pt"
            _save_sb3_weights(model, ckpt)
            print(f"  Saved {ckpt.name}")

            if cb.check(model, steps_this_iter=50_000):
                print(f"  → Stage {stage_name} passed! Advancing.")
                break

        vec_env.close()

        # If stage not passed: save current weights and STOP — no point training harder stages
        if not cb.stage_passed:
            best_rank = min(cb._history) if cb._history else float("nan")
            print(
                f"\n  ✗ Stage {stage_name} failed — best avg_rank {best_rank:.2f} "
                f"never reached threshold {wr_thresh:.1f} within {steps_done:,} steps.\n"
                f"  Stopping curriculum. Current checkpoint is the best available."
            )
            all_stages_passed = False
            break  # do NOT advance to harder stages

    _save_sb3_weights(model, best_ckpt)
    status = "all stages passed ✓" if all_stages_passed else f"stopped at stage {stage_idx} ({stage_name}) ✗"
    print(f"\nFinal curriculum checkpoint: {best_ckpt.name}  [{status}]")
    return best_ckpt


def train_self_play(
    output_dir: Path,
    snapshot_dir: Path,
    total_steps: int = 1_000_000,
    n_envs: int = 8,
    snapshot_every: int = 50_000,
    init_from: Path | None = None,
    device: str = "auto",
) -> Path:
    """Phase 4: continuous self-play PPO training."""
    import warnings
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv

    warnings.filterwarnings("ignore", message=".*shared layers.*", category=UserWarning)
    MaskablePPO, _ = _make_maskable_ppo()
    _write_past_agent_wrapper()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models.policy_network import BomberCNNExtractor

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [_make_self_play_env_fn(snapshot_dir, seed=i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs={
            "features_extractor_class": BomberCNNExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[128], vf=[128]),
        },
        verbose=1,
        device=device,
        tensorboard_log=str(output_dir / "tb_logs"),
        **PPO_DEFAULTS,
    )

    if init_from and init_from.exists():
        _load_bc_into_sb3(model, init_from, device)

    steps_done = 0
    snap_count = 0
    best_ckpt = output_dir / "selfplay_best.pt"

    while steps_done < total_steps:
        model.learn(total_timesteps=snapshot_every, reset_num_timesteps=False)
        steps_done += snapshot_every
        snap_count += 1

        ts = time.strftime("%Y%m%d_%H%M%S")
        snap_path = snapshot_dir / f"snapshot_{steps_done}steps_{ts}.pt"
        _save_sb3_weights(model, snap_path)
        _save_sb3_weights(model, best_ckpt)

        # Prune old snapshots (keep latest 20)
        all_snaps = sorted(snapshot_dir.glob("snapshot_*.pt"), key=lambda p: p.stat().st_mtime)
        for old in all_snaps[:-20]:
            old.unlink(missing_ok=True)

        print(f"Step {steps_done}/{total_steps} — snapshot {snap_count} saved")

    vec_env.close()
    return best_ckpt


def _save_sb3_weights(model, path: Path) -> None:
    """Extract BomberPolicyNet-compatible weights from SB3 model and save.

    Mapping (SB3 MultiInputPolicy → BomberPolicyNet):
      features_extractor.spatial_enc  → spatial_enc
      features_extractor.aux_enc      → aux_enc
      features_extractor.fusion[0]    → fusion[0]  (Linear 3168→256)
      mlp_extractor.policy_net[0]     → fusion[2]  (Linear 256→128)
      action_net                       → policy_head (Linear 128→6)
      value_net                        → value_head  (Linear 128→1)
    """
    import torch
    from src.models.policy_network import BomberPolicyNet

    net = BomberPolicyNet()
    fe  = model.policy.features_extractor
    pol = model.policy

    try:
        net.spatial_enc.load_state_dict(fe.spatial_enc.state_dict())
        net.aux_enc.load_state_dict(fe.aux_enc.state_dict())
        # fusion[0] = Linear(3168→256), fusion[1] = ReLU (no params)
        net.fusion[0].load_state_dict(fe.fusion[0].state_dict())
        # fusion[2] = Linear(256→128) comes from SB3 mlp_extractor.policy_net[0]
        net.fusion[2].load_state_dict(pol.mlp_extractor.policy_net[0].state_dict())
        net.policy_head.load_state_dict(pol.action_net.state_dict())
        net.value_head.load_state_dict(pol.value_net.state_dict())
    except Exception as e:
        print(f"  Warning: partial weight transfer ({e})")

    torch.save({"model_state_dict": net.state_dict()}, path)


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _cli() -> None:
    parser = argparse.ArgumentParser(description="PPO trainer for BomberPolicyNet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--curriculum",  action="store_true")
    group.add_argument("--self-play",   action="store_true")
    parser.add_argument("--output-dir",    type=Path, default=_ROOT / "checkpoints")
    parser.add_argument("--snapshot-dir",  type=Path, default=_ROOT / "checkpoints" / "past_agents")
    parser.add_argument("--init-from",     type=Path, default=None)
    parser.add_argument("--total-steps",   type=int,  default=500_000)
    parser.add_argument("--n-envs",        type=int,  default=8)
    parser.add_argument("--device",        type=str,  default="auto")
    args = parser.parse_args()

    if args.curriculum:
        train_curriculum(
            output_dir=args.output_dir,
            total_steps_per_stage=args.total_steps,
            n_envs=args.n_envs,
            init_from=args.init_from,
            device=args.device,
        )
    else:
        train_self_play(
            output_dir=args.output_dir,
            snapshot_dir=args.snapshot_dir,
            total_steps=args.total_steps,
            n_envs=args.n_envs,
            init_from=args.init_from,
            device=args.device,
        )


if __name__ == "__main__":
    _cli()
