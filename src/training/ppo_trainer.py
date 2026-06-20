from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import os
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_maskable_ppo():
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        return MaskablePPO, MaskableEvalCallback
    except ImportError as e:
        raise ImportError(
            "sb3-contrib not found. Install with: pip install sb3-contrib"
        ) from e


PPO_DEFAULTS: dict = {
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      256,
    "n_epochs":        10,
    "gamma":           0.995,
    "gae_lambda":      0.95,
    "clip_range":      0.3,
    "ent_coef":        0.03,
    "vf_coef":         0.5,
    "max_grad_norm":   0.5,
}

CURRICULUM_STAGES = [
    ("random",          0.8),
    ("simple",          1.5),
    ("simple_smarter1", 1.5),
    ("smarter",         1.8),
    ("tactical",        2.0),
    ("trapper",         2.0),
    ("genius",          2.0),
]

STAGE_ENT_COEF = [0.08, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]

STAGE_LR = [3e-4, 1.5e-4, 1.2e-4, 1e-4, 8e-5, 8e-5, 5e-5]

MIN_STEPS_PER_STAGE = 100_000


def _make_opponents(stage_name: str, opp_ids: list[int], mix_random: bool = False,
                    stage_idx: int = 0):
    from agent import (
        GeniusRuleAgent, TacticalRuleAgent, TrapperRuleAgent,
        SmarterRuleAgent, SimpleRuleAgent, RandomAgent,
    )

    if stage_name == "random":
        return [RandomAgent(i) for i in opp_ids]
    if stage_name == "simple":
        opps = [SimpleRuleAgent(i) for i in opp_ids]
    elif stage_name == "simple_smarter1":
        opps = [SmarterRuleAgent(opp_ids[0]), SimpleRuleAgent(opp_ids[1]), SimpleRuleAgent(opp_ids[2])]
    elif stage_name == "smarter":
        opps = [SmarterRuleAgent(opp_ids[0]), SmarterRuleAgent(opp_ids[1]), SimpleRuleAgent(opp_ids[2])]
    elif stage_name == "tactical":
        opps = [TacticalRuleAgent(opp_ids[0]), TacticalRuleAgent(opp_ids[1]), SmarterRuleAgent(opp_ids[2])]
    elif stage_name == "trapper":
        opps = [TrapperRuleAgent(opp_ids[0]), TrapperRuleAgent(opp_ids[1]), TacticalRuleAgent(opp_ids[2])]
    elif stage_name == "genius":
        opps = [GeniusRuleAgent(opp_ids[0]), GeniusRuleAgent(opp_ids[1]), TacticalRuleAgent(opp_ids[2])]
    else:
        raise ValueError(f"Unknown stage: {stage_name}")

    if mix_random and random.random() < 0.20:
        mix_slot = 0 if stage_idx >= 4 else len(opps) - 1
        opps[mix_slot] = RandomAgent(opp_ids[mix_slot])

    return opps


def _make_env_fn(stage_name: str, seed: int, agent_id: int = 0, mix_random: bool = False,
                 stage_idx: int = 0) -> Callable:
    def _factory():
        from src.wrappers.bomberland_env import SingleAgentBomberEnv
        all_ids = list(range(4))
        all_ids.remove(agent_id)
        opps = _make_opponents(stage_name, all_ids, mix_random=mix_random, stage_idx=stage_idx)
        env = SingleAgentBomberEnv(opponents=opps, agent_id=agent_id, seed=seed)
        return env
    return _factory


def _compute_pfsp_weights(
    snapshots: list,
    win_rate_table: dict,
    pfsp_temperature: float = 1.0,
    recency_alpha: float = 0.9,
    pfsp_blend: float = 0.7,
) -> np.ndarray:
    n = len(snapshots)
    if n == 0:
        return np.array([], dtype=np.float64)

    pfsp_raw = np.array(
        [1.0 - win_rate_table.get(str(s), 0.5) for s in snapshots],
        dtype=np.float64,
    )
    pfsp_raw = np.power(pfsp_raw, 1.0 / max(pfsp_temperature, 1e-6))
    pfsp_sum = pfsp_raw.sum()
    pfsp_w = pfsp_raw / pfsp_sum if pfsp_sum > 0 else np.ones(n) / n

    recency_raw = np.array([recency_alpha ** i for i in range(n - 1, -1, -1)], dtype=np.float64)
    recency_sum = recency_raw.sum()
    recency_w = recency_raw / recency_sum if recency_sum > 0 else np.ones(n) / n

    combined = pfsp_blend * pfsp_w + (1.0 - pfsp_blend) * recency_w
    combined_sum = combined.sum()
    return combined / combined_sum if combined_sum > 0 else np.ones(n) / n


def _make_self_play_env_fn(
    snapshot_dir: Path,
    seed: int,
    agent_id: int = 0,
    pool_ref: list | None = None,
    win_rate_table_ref: dict | None = None,
) -> Callable:
    _pool: list = pool_ref if pool_ref is not None else []
    _win_rate_table: dict = win_rate_table_ref if win_rate_table_ref is not None else {}

    def _factory():
        from src.wrappers.bomberland_env import SingleAgentBomberEnv

        all_ids = list(range(4))
        all_ids.remove(agent_id)

        snapshots = list(_pool) if _pool else sorted(snapshot_dir.glob("*.pt"))

        if not snapshots:
            from agent import GeniusRuleAgent, TacticalRuleAgent, TrapperRuleAgent
            opps = [TrapperRuleAgent(all_ids[0]), TacticalRuleAgent(all_ids[1]), GeniusRuleAgent(all_ids[2])]
        else:
            from src.inference.past_agent import PastAgentWrapper
            if _win_rate_table:
                weights = _compute_pfsp_weights(snapshots, _win_rate_table)
            else:
                weights = np.array([0.9 ** i for i in range(len(snapshots) - 1, -1, -1)])
                weights /= weights.sum()
            chosen = np.random.choice(snapshots, size=3, replace=True, p=weights)
            opps = [PastAgentWrapper(str(c), i) for c, i in zip(chosen, all_ids)]

        env = SingleAgentBomberEnv(opponents=opps, agent_id=agent_id, seed=seed)
        return env
    return _factory


def _set_lr(model, lr: float) -> None:
    model.learning_rate = lr
    model.lr_schedule = lambda _: lr
    for group in model.policy.optimizer.param_groups:
        group["lr"] = lr
    model.policy.optimizer.state.clear()
    print(f"  Optimizer lr={lr:.1e}, Adam state cleared")


def _load_bc_into_sb3(model, bc_path: Path, device: str) -> None:
    import torch
    from src.models.policy_network import BomberPolicyNet

    bc_net = BomberPolicyNet.load(str(bc_path), device=device)
    fe  = model.policy.features_extractor
    pol = model.policy

    fe.spatial_enc.load_state_dict(bc_net.spatial_enc.state_dict())
    fe.aux_enc.load_state_dict(bc_net.aux_enc.state_dict())
    fe.fusion.load_state_dict(bc_net.fusion.state_dict())
    pol.mlp_extractor.policy_net[0].load_state_dict(bc_net.policy_head[0].state_dict())
    pol.mlp_extractor.value_net[0].load_state_dict(bc_net.value_head[0].state_dict())
    pol.action_net.load_state_dict(bc_net.policy_head[2].state_dict())
    pol.value_net.load_state_dict(bc_net.value_head[2].state_dict())
    print(f"  Loaded BC weights (all layers) from {bc_path.name}")


def _write_past_agent_wrapper():
    dest = _ROOT / "src" / "inference" / "past_agent.py"
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.parent.joinpath("__init__.py").touch()
    dest.write_text(
        '''from __future__ import annotations
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
    def __init__(self, ckpt_path: str, agent_id: int) -> None:
        self.agent_id = int(agent_id)
        self._net = BomberPolicyNet.load(ckpt_path, device="cpu")
        self._step: int = 0
        self._initial_boxes: int = 50
        self._my_kills: int = 0
        self._my_boxes: int = 0
        self._prev_players = None
        self._prev_grid: "np.ndarray | None" = None

    def reset(self) -> None:
        self._step = 0
        self._initial_boxes = 50
        self._my_kills = 0
        self._my_boxes = 0
        self._prev_players = None
        self._prev_grid = None

    def act(self, obs: dict) -> int:
        curr_players = np.asarray(obs["players"])
        curr_grid    = np.asarray(obs["map"], dtype=np.int8)

        if self._prev_players is not None:
            prev_en = sum(int(self._prev_players[i][2]) for i in range(4) if i != self.agent_id)
            curr_en = sum(int(curr_players[i][2])       for i in range(4) if i != self.agent_id)
            self._my_kills += max(0, prev_en - curr_en)
        if self._prev_grid is not None:
            self._my_boxes += int(((self._prev_grid == 2) & (curr_grid != 2)).sum())
        if self._step == 0:
            self._initial_boxes = max(1, int((curr_grid == 2).sum()))

        boxes_now = int((curr_grid == 2).sum())

        spatial, aux = extract_features(
            obs, self.agent_id,
            step=self._step,
            initial_boxes=self._initial_boxes,
            boxes_remaining=boxes_now,
            my_kills=self._my_kills,
            my_boxes_destroyed=self._my_boxes,
        )
        mask = compute_action_mask(obs, self.agent_id)
        sp_t = torch.from_numpy(spatial).unsqueeze(0)
        ax_t = torch.from_numpy(aux).unsqueeze(0)
        with torch.no_grad():
            logits = self._net.get_action_logits(sp_t, ax_t).squeeze(0).numpy()
        masked = apply_mask_to_logits(logits, mask)

        self._step += 1
        self._prev_players = curr_players.copy()
        self._prev_grid    = curr_grid.copy()

        return int(np.argmax(masked))
'''
    )


class CurriculumAdvanceCallback:
    def __init__(
        self,
        eval_env_fn: Callable,
        n_eval_episodes: int = 200,
        rank_threshold: float = 1.5,
        patience: int = 3,
        min_steps: int = MIN_STEPS_PER_STAGE,
    ) -> None:
        self.eval_env_fn = eval_env_fn
        self.n_eval_episodes = n_eval_episodes
        self.threshold = rank_threshold
        self.patience = patience
        self.min_steps = min_steps
        self._history: list[float] = []
        self._steps_in_stage: int = 0
        self.stage_passed: bool = False

    def evaluate(self, model) -> tuple[float, dict[str, float]]:
        env = self.eval_env_fn()
        total_rank = 0.0
        component_sums: dict[str, float] = {}
        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            final_info: dict = {}
            ep_components: dict[str, float] = {}
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, terminated, truncated, final_info = env.step(int(action))
                done = terminated or truncated
                step_reward_info = final_info.get("reward_info", {})
                for k, v in step_reward_info.items():
                    ep_components[k] = ep_components.get(k, 0.0) + float(v)
            raw = env.raw_obs
            if raw is not None:
                alive = [int(raw["players"][i][2]) for i in range(4)]
                n_alive = sum(alive)
                if not alive[0]:
                    rank = 3.0
                elif n_alive == 1:
                    rank = 0.0
                else:
                    agent_kills = final_info.get("kills", 0)
                    if agent_kills > 0:
                        rank = 0.0
                    else:
                        rank = (n_alive - 1) / 2.0
            else:
                rank = 3.0
            total_rank += rank
            for k, v in ep_components.items():
                component_sums[k] = component_sums.get(k, 0.0) + v
        env.close()
        n = max(self.n_eval_episodes, 1)
        reward_component_avgs = {k: v / n for k, v in component_sums.items()}
        return total_rank / n, reward_component_avgs

    def check(self, model, steps_this_iter: int = 50_000) -> bool:
        self._steps_in_stage += steps_this_iter
        avg_rank, reward_component_avgs = self.evaluate(model)
        self._history.append(avg_rank)
        print(f"  Avg rank: {avg_rank:.2f} (threshold <= {self.threshold:.1f})")
        try:
            model.logger.record("eval/avg_rank", avg_rank)
            for component_name, component_mean in reward_component_avgs.items():
                model.logger.record(f"eval/reward_{component_name}", component_mean)
            model.logger.dump(step=model.num_timesteps)
        except Exception:
            pass

        if self._steps_in_stage < self.min_steps:
            print(f"  Min steps not reached ({self._steps_in_stage:,}/{self.min_steps:,}), holding")
            return False

        if len(self._history) >= 3:
            rolling = sum(self._history[-3:]) / 3
            print(f"  Rolling mean (last 3): {rolling:.3f}")
            if rolling <= self.threshold:
                self.stage_passed = True
                return True

        return False


def train_curriculum(
    output_dir: Path,
    total_steps_per_stage: int = 750_000,
    n_envs: int = 8,
    init_from: Path | None = None,
    init_from_tactical: Path | None = None,
    device: str = "auto",
    log_dir: Path | None = None,
    eval_freq: int = 50_000,
    eval_episodes: int = 200,
    min_steps_per_stage: int = MIN_STEPS_PER_STAGE,
) -> tuple[Path, bool]:
    import warnings
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    warnings.filterwarnings("ignore", message=".*shared layers.*", category=UserWarning)

    MaskablePPO, _ = _make_maskable_ppo()
    _write_past_agent_wrapper()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models.policy_network import BomberCNNExtractor, make_observation_space

    output_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = str(log_dir) if log_dir is not None else str(output_dir / "tb_logs")
    best_ckpt = output_dir / "ppo_curriculum_best.pt"

    model = None
    all_stages_passed = True
    vec_env = None

    for stage_idx, (stage_name, wr_thresh) in enumerate(CURRICULUM_STAGES):
        print(f"\n=== Curriculum Stage {stage_idx}: {stage_name} (avg_rank threshold <= {wr_thresh:.1f}) ===")

        env_fns = [
            _make_env_fn(stage_name, seed=stage_idx * 1000 + i, mix_random=True, stage_idx=stage_idx)
            for i in range(n_envs)
        ]
        vecnorm_path = output_dir / f"vecnormalize_{stage_idx}.pkl"
        raw_vec_env = SubprocVecEnv(env_fns)
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), raw_vec_env)
            vec_env.training = True
            print(f"  Loaded VecNormalize stats from {vecnorm_path.name}")
        else:
            prev_vecnorm_path = output_dir / f"vecnormalize_{stage_idx - 1}.pkl"
            if stage_idx > 0 and prev_vecnorm_path.exists():
                vec_env = VecNormalize.load(str(prev_vecnorm_path), raw_vec_env)
                vec_env.training = True
                print(f"  Warm-started VecNormalize stats from stage {stage_idx - 1} ({prev_vecnorm_path.name})")
            else:
                vec_env = VecNormalize(
                    raw_vec_env,
                    norm_obs=False,
                    norm_reward=True,
                    clip_reward=10.0,
                    gamma=PPO_DEFAULTS["gamma"],
                )

        if model is None:
            model = MaskablePPO(
                "MultiInputPolicy",
                vec_env,
                policy_kwargs={
                    "features_extractor_class": BomberCNNExtractor,
                    "features_extractor_kwargs": {"features_dim": 256},
                    "net_arch": dict(pi=[256], vf=[256]),
                },
                verbose=1,
                device=device,
                tensorboard_log=tb_log_dir,
                **PPO_DEFAULTS,
            )
            if init_from and init_from.exists():
                _load_bc_into_sb3(model, init_from, device)
            model.ent_coef = STAGE_ENT_COEF[stage_idx]
            _set_lr(model, STAGE_LR[stage_idx])
            print(f"  ent_coef={STAGE_ENT_COEF[stage_idx]}, lr={STAGE_LR[stage_idx]:.1e} for stage {stage_idx}")
        else:
            model.set_env(vec_env)
            model.ent_coef = STAGE_ENT_COEF[stage_idx]
            _set_lr(model, STAGE_LR[stage_idx])
            print(f"  ent_coef={STAGE_ENT_COEF[stage_idx]}, lr={STAGE_LR[stage_idx]:.1e} (reset) for stage {stage_idx}")

        eval_fn = _make_env_fn(stage_name, seed=9999, mix_random=False, stage_idx=stage_idx)
        cb = CurriculumAdvanceCallback(
            eval_fn,
            n_eval_episodes=eval_episodes,
            rank_threshold=wr_thresh,
            patience=3,
            min_steps=min_steps_per_stage,
        )

        stage_best_rank = float("inf")
        stage_best_ckpt = output_dir / f"ppo_s{stage_idx}_best.pt"

        stage_start_timesteps: int = model.num_timesteps
        steps_done = 0
        while steps_done < total_steps_per_stage:
            model.learn(
                total_timesteps=eval_freq,
                reset_num_timesteps=False,
                tb_log_name=f"stage_{stage_idx}_{stage_name}",
            )
            _steps_in_stage = model.num_timesteps - stage_start_timesteps
            steps_done = _steps_in_stage

            ts = time.strftime("%Y%m%d_%H%M%S")
            ckpt = output_dir / f"ppo_s{stage_idx}_{steps_done}steps_{ts}.pt"
            _save_sb3_weights(model, ckpt)
            print(f"  Saved {ckpt.name}")

            advanced = cb.check(model, steps_this_iter=_steps_in_stage - cb._steps_in_stage)

            last_rank = cb._history[-1]
            if last_rank < stage_best_rank:
                stage_best_rank = last_rank
                _save_sb3_weights(model, stage_best_ckpt)
                print(f"  Stage best: avg_rank {stage_best_rank:.2f} -> {stage_best_ckpt.name}")

            if advanced:
                print(f"  Stage {stage_name} passed! Advancing.")
                break

        vec_env.save(str(vecnorm_path))
        vec_env.close()

        if stage_best_ckpt.exists():
            shutil.copy2(stage_best_ckpt, best_ckpt)
            if cb.stage_passed:
                if stage_idx == 0:
                    _anchor_ckpt = (
                        init_from_tactical if (init_from_tactical and init_from_tactical.exists())
                        else init_from if (init_from and init_from.exists())
                        else None
                    )
                    if _anchor_ckpt is not None:
                        _load_bc_into_sb3(model, _anchor_ckpt, device)
                        print(f"  Re-anchored to {_anchor_ckpt.name} for Stage 1")
                else:
                    _restore_sb3_weights(model, stage_best_ckpt, device)

        if not cb.stage_passed:
            best_rank = min(cb._history) if cb._history else float("nan")
            print(
                f"\n  Stage {stage_name} failed — best avg_rank {best_rank:.2f} "
                f"never reached threshold {wr_thresh:.1f} within {steps_done:,} steps.\n"
                f"  Stopping curriculum."
            )
            all_stages_passed = False
            break

    status = "all stages passed" if all_stages_passed else f"stopped at stage {stage_idx} ({stage_name})"
    print(f"\nFinal curriculum checkpoint: {best_ckpt.name}  [{status}]")
    return best_ckpt, all_stages_passed


def train_self_play(
    output_dir: Path,
    snapshot_dir: Path,
    total_steps: int = 500_000,
    n_envs: int = 8,
    snapshot_every: int = 50_000,
    init_from: Path | None = None,
    device: str = "auto",
    log_dir: Path | None = None,
    pool_size: int = 20,
) -> Path:
    import warnings
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    warnings.filterwarnings("ignore", message=".*shared layers.*", category=UserWarning)
    MaskablePPO, _ = _make_maskable_ppo()
    _write_past_agent_wrapper()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models.policy_network import BomberCNNExtractor

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = str(log_dir) if log_dir is not None else str(output_dir / "tb_logs")

    pool_ref: list = []
    win_rate_table: dict[str, float] = {}

    env_fns = [
        _make_self_play_env_fn(
            snapshot_dir, seed=i, pool_ref=pool_ref, win_rate_table_ref=win_rate_table
        )
        for i in range(n_envs)
    ]
    raw_vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(
        raw_vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=PPO_DEFAULTS["gamma"],
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs={
            "features_extractor_class": BomberCNNExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[256], vf=[256]),
        },
        verbose=1,
        device=device,
        tensorboard_log=tb_log_dir,
        **PPO_DEFAULTS,
    )

    if init_from and init_from.exists():
        _load_bc_into_sb3(model, init_from, device)

    _set_lr(model, STAGE_LR[-1])
    model.ent_coef = STAGE_ENT_COEF[-1]
    print(f"  Self-play init: ent_coef={STAGE_ENT_COEF[-1]}, lr={STAGE_LR[-1]:.1e}")

    steps_done = 0
    snap_count = 0
    best_ckpt = output_dir / "selfplay_best.pt"
    best_selfplay_rank: float = float("inf")

    eval_env_fn = _make_env_fn("genius", seed=9999, mix_random=False, stage_idx=6)

    def _eval_vs_genius(n_episodes: int = 100) -> float:
        env = eval_env_fn()
        total_rank = 0.0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            final_info: dict = {}
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, terminated, truncated, final_info = env.step(int(action))
                done = terminated or truncated
            raw = env.raw_obs
            if raw is not None:
                alive = [int(raw["players"][i][2]) for i in range(4)]
                n_alive = sum(alive)
                if not alive[0]:
                    rank = 3.0
                elif n_alive == 1:
                    rank = 0.0
                else:
                    agent_kills = final_info.get("kills", 0)
                    rank = 0.0 if agent_kills > 0 else (n_alive - 1) / 2.0
            else:
                rank = 3.0
            total_rank += rank
        env.close()
        return total_rank / n_episodes

    def _eval_win_rate_vs_snapshot(snap_path: Path, n_episodes: int = 30) -> float:
        from src.inference.past_agent import PastAgentWrapper
        from src.wrappers.bomberland_env import SingleAgentBomberEnv

        all_ids = list(range(4))
        all_ids.remove(0)
        try:
            opps = [PastAgentWrapper(str(snap_path), i) for i in all_ids]
            env = SingleAgentBomberEnv(opponents=opps, agent_id=0, seed=42)
        except Exception:
            return 0.5

        wins = 0.0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            final_info: dict = {}
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, terminated, truncated, final_info = env.step(int(action))
                done = terminated or truncated
            raw = env.raw_obs
            if raw is not None:
                alive = [int(raw["players"][i][2]) for i in range(4)]
                n_alive = sum(alive)
                if alive[0] and n_alive == 1:
                    wins += 1.0
                elif alive[0]:
                    agent_kills = final_info.get("kills", 0)
                    wins += 0.5 if agent_kills > 0 else 0.0
        env.close()
        return wins / max(n_episodes, 1)

    while steps_done < total_steps:
        model.learn(total_timesteps=snapshot_every, reset_num_timesteps=False)
        steps_done += snapshot_every
        snap_count += 1

        ts = time.strftime("%Y%m%d_%H%M%S")
        snap_path = snapshot_dir / f"snapshot_{steps_done}steps_{ts}.pt"
        _save_sb3_weights(model, snap_path)

        all_snaps = sorted(snapshot_dir.glob("snapshot_*.pt"), key=lambda p: p.stat().st_mtime)
        for old in all_snaps[:-pool_size]:
            old.unlink(missing_ok=True)
        kept_snaps = sorted(snapshot_dir.glob("snapshot_*.pt"), key=lambda p: p.stat().st_mtime)
        pool_ref.clear()
        pool_ref.extend(kept_snaps)

        avg_rank = _eval_vs_genius(n_episodes=100)
        print(f"Step {steps_done}/{total_steps} | snapshot {snap_count} | avg_rank vs genius: {avg_rank:.3f}")
        if avg_rank < best_selfplay_rank:
            best_selfplay_rank = avg_rank
            _save_sb3_weights(model, best_ckpt)
            print(f"  New best selfplay: avg_rank {best_selfplay_rank:.3f} -> {best_ckpt.name}")
        else:
            print(f"  No improvement (best={best_selfplay_rank:.3f}); timestamped snapshot saved only")

        if kept_snaps:
            print(f"  PFSP: evaluating win rates against {len(kept_snaps)} pool snapshots ...")
            live_keys = {str(s) for s in kept_snaps}
            stale = [k for k in list(win_rate_table.keys()) if k not in live_keys]
            for k in stale:
                del win_rate_table[k]

            pfsp_log_lines: list[str] = []
            for snap in kept_snaps:
                wr = _eval_win_rate_vs_snapshot(snap, n_episodes=30)
                win_rate_table[str(snap)] = wr
                pfsp_log_lines.append(f"{snap.name}: win_rate={wr:.2f}")

            pfsp_weights = _compute_pfsp_weights(kept_snaps, win_rate_table)
            for snap, w, line in zip(kept_snaps, pfsp_weights, pfsp_log_lines):
                print(f"    {line} | pfsp_weight={w:.3f}")
            try:
                mean_win_rate = float(np.mean(list(win_rate_table.values())))
                model.logger.record("self_play/pfsp_mean_win_rate", mean_win_rate)
                model.logger.record("self_play/pfsp_mean_loss_rate", 1.0 - mean_win_rate)
                model.logger.dump(step=model.num_timesteps)
            except Exception:
                pass

    vec_env.close()
    return best_ckpt


def _save_sb3_weights(model, path: Path) -> None:
    import torch
    from src.models.policy_network import BomberPolicyNet

    net = BomberPolicyNet()
    fe  = model.policy.features_extractor
    pol = model.policy

    try:
        net.spatial_enc.load_state_dict(fe.spatial_enc.state_dict())
        net.aux_enc.load_state_dict(fe.aux_enc.state_dict())
        net.fusion.load_state_dict(fe.fusion.state_dict())
        net.policy_head[0].load_state_dict(pol.mlp_extractor.policy_net[0].state_dict())
        net.policy_head[2].load_state_dict(pol.action_net.state_dict())
        net.value_head[0].load_state_dict(pol.mlp_extractor.value_net[0].state_dict())
        net.value_head[2].load_state_dict(pol.value_net.state_dict())
    except Exception as e:
        print(f"  Warning: partial weight transfer ({e})")

    torch.save({"model_state_dict": net.state_dict()}, path)


def _restore_sb3_weights(model, checkpoint_path: Path, device: str) -> None:
    import torch
    from src.models.policy_network import BomberPolicyNet

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net = BomberPolicyNet()
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device)

    fe  = model.policy.features_extractor
    pol = model.policy

    try:
        fe.spatial_enc.load_state_dict(net.spatial_enc.state_dict())
        fe.aux_enc.load_state_dict(net.aux_enc.state_dict())
        fe.fusion.load_state_dict(net.fusion.state_dict())
        pol.mlp_extractor.policy_net[0].load_state_dict(net.policy_head[0].state_dict())
        pol.mlp_extractor.value_net[0].load_state_dict(net.value_head[0].state_dict())
        pol.action_net.load_state_dict(net.policy_head[2].state_dict())
        pol.value_net.load_state_dict(net.value_head[2].state_dict())
        print(f"  Restored stage-best weights from {checkpoint_path.name}")
    except Exception as e:
        print(f"  Warning: partial weight restore ({e})")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="PPO trainer for BomberPolicyNet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--curriculum",  action="store_true")
    group.add_argument("--self-play",   action="store_true")
    parser.add_argument("--output-dir",              type=Path, default=_ROOT / "checkpoints")
    parser.add_argument("--log-dir",                 type=Path, default=_ROOT / "logs")
    parser.add_argument("--snapshot-dir",            type=Path, default=_ROOT / "checkpoints" / "past_agents")
    parser.add_argument("--init-from",               type=Path, default=None)
    parser.add_argument("--init-from-tactical",      type=Path, default=None)
    parser.add_argument("--total-steps-per-stage",   type=int,  default=None)
    parser.add_argument("--total-steps",             type=int,  default=500_000)
    parser.add_argument("--n-envs",                  type=int,  default=8)
    parser.add_argument("--eval-freq",               type=int,  default=50_000)
    parser.add_argument("--eval-episodes",           type=int,  default=200)
    parser.add_argument("--min-steps-per-stage",     type=int,  default=MIN_STEPS_PER_STAGE)
    parser.add_argument("--snapshot-every",          type=int,  default=50_000)
    parser.add_argument("--pool-size",               type=int,  default=20)
    parser.add_argument("--device",                  type=str,  default="auto")
    args = parser.parse_args()

    steps_per_stage = args.total_steps_per_stage if args.total_steps_per_stage is not None else 750_000

    if args.curriculum:
        train_curriculum(
            output_dir=args.output_dir,
            total_steps_per_stage=steps_per_stage,
            n_envs=args.n_envs,
            init_from=args.init_from,
            init_from_tactical=args.init_from_tactical,
            device=args.device,
            log_dir=args.log_dir if args.log_dir != _ROOT / "logs" else None,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            min_steps_per_stage=args.min_steps_per_stage,
        )
    else:
        train_self_play(
            output_dir=args.output_dir,
            snapshot_dir=args.snapshot_dir,
            total_steps=args.total_steps,
            n_envs=args.n_envs,
            snapshot_every=args.snapshot_every,
            init_from=args.init_from,
            device=args.device,
            log_dir=args.log_dir if args.log_dir != _ROOT / "logs" else None,
            pool_size=args.pool_size,
        )


if __name__ == "__main__":
    _cli()
