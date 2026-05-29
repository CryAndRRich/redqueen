"""
Tactical BC — replace history-game Behavioral Cloning with TacticalRuleAgent self-rollout BC.

Two modes (select via CLI flags):
  --generate  Run N full 4-player games of TacticalRuleAgent vs itself, collect winner
              demonstrations, save to <data-dir>/tactical_bc_dataset.npz
  --train     Train BomberPolicyNet (focal loss gamma=2) on the generated dataset,
              save periodic + best checkpoints to checkpoints/

Usage examples:
  python -m src.training.tactical_bc --generate --n-games 200
  python -m src.training.tactical_bc --train --epochs 20
  python -m src.training.tactical_bc --generate --train --n-games 500 --epochs 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — allow running as both module and script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.logic.action_masking import compute_action_mask  # noqa: E402
from src.models.policy_network import BomberPolicyNet  # noqa: E402
from src.utils.feature_extractor import count_boxes, extract_features  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_AGENTS: int = 4
N_ACTIONS: int = 6


# ===========================================================================
# Data generation — TacticalRuleAgent self-rollout
# ===========================================================================

def _determine_winner(ranks: list[int], survival_steps: list[int]) -> Optional[int]:
    """
    Return the agent index with rank == 0 (the winner).
    If no agent achieved rank 0 (tie at step 500) return the agent with the
    lowest rank; break ties by longest survival then lowest agent_id.
    """
    for i, r in enumerate(ranks):
        if r == 0:
            return i
    # Fallback: pick best-ranked survivor
    best_rank = min(ranks)
    candidates = [i for i, r in enumerate(ranks) if r == best_rank]
    candidates.sort(key=lambda i: (-survival_steps[i], i))
    return candidates[0] if candidates else 0


def _rank_players(players_list: list) -> list[int]:
    """
    Assign ranks [0..3] to each player from the engine Player objects.
    Rank 0 = winner. Tie-break: kills > boxes > items > bombs > agent_id.
    """
    n = len(players_list)
    order = sorted(
        range(n),
        key=lambda i: (
            -players_list[i].stats["kills"],
            -players_list[i].stats["boxes"],
            -players_list[i].stats["items"],
            -players_list[i].stats["bombs"],
            i,
        ),
    )
    ranks = [0] * n
    for rank_pos, agent_i in enumerate(order):
        ranks[agent_i] = rank_pos
    return ranks


def generate_dataset(
    n_games: int,
    output_path: Path,
    seed: Optional[int] = None,
) -> None:
    """
    Run n_games of 4x TacticalRuleAgent, collect winner demonstrations,
    and save the dataset to output_path as a .npz file.

    Args:
        n_games:     Number of full games to simulate.
        output_path: Where to write the .npz dataset file.
        seed:        Optional base RNG seed (each game gets seed+game_idx).
    """
    # Lazy import engine and agent — avoids hard dependency at module level
    from agent.tactical_rule_agent import TacticalRuleAgent  # noqa: E402
    from engine.game import BomberEnv  # noqa: E402

    rng_base = seed if seed is not None else int(time.time()) & 0xFFFFFF

    all_spatial: list[np.ndarray] = []
    all_aux: list[np.ndarray] = []
    all_actions: list[int] = []
    all_masks: list[np.ndarray] = []

    total_demos = 0

    for game_idx in tqdm(range(n_games), desc="Generating games"):
        game_seed = rng_base + game_idx
        env = BomberEnv(seed=game_seed)
        obs = env.reset(seed=game_seed)

        agents = [TacticalRuleAgent(i) for i in range(N_AGENTS)]

        # Per-game history buffer: (obs_before_action, action_taken) for each agent
        # We only commit the winner's transitions after the game ends.
        history: list[dict] = []
        # Track survival step for each agent
        survival_steps = [0] * N_AGENTS
        initial_boxes = count_boxes(obs["map"])

        terminated = False
        truncated = False
        step = 0

        while not terminated and not truncated:
            # Compute actions for all alive agents
            actions_this_step: list[int] = []
            alive_flags = [int(obs["players"][i][2]) for i in range(N_AGENTS)]
            step_record: dict = {
                "obs": obs,
                "step": step,
                "initial_boxes": initial_boxes,
                "alive_before": list(alive_flags),
            }
            agent_actions: list[int] = []
            for i in range(N_AGENTS):
                if alive_flags[i] == 1:
                    action = int(agents[i].act(obs))
                    survival_steps[i] = step
                else:
                    action = 0  # dead agent always STOP
                agent_actions.append(action)
            step_record["actions"] = list(agent_actions)
            history.append(step_record)

            obs, terminated, truncated = env.step(agent_actions)
            step += 1

        # Final survival step update
        final_alive = [int(obs["players"][i][2]) for i in range(N_AGENTS)]
        for i in range(N_AGENTS):
            if final_alive[i] == 1:
                survival_steps[i] = step

        # Determine winner from engine Player stats
        ranks = _rank_players(env.players)
        winner_id = _determine_winner(ranks, survival_steps)

        if winner_id is None:
            continue  # skip degenerate game

        # Collect winner's transitions from history
        winner_demos = 0
        for record in history:
            agent_id = winner_id
            if record["alive_before"][agent_id] != 1:
                continue  # agent was already dead this step

            action_taken = record["actions"][agent_id]
            frame_obs = record["obs"]
            frame_step = record["step"]
            frame_initial_boxes = record["initial_boxes"]

            spatial, aux = extract_features(
                obs=frame_obs,
                agent_id=agent_id,
                step=frame_step,
                total_steps=500,
                initial_boxes=frame_initial_boxes,
                boxes_remaining=None,
                my_kills=0,
                my_boxes_destroyed=0,
            )
            action_mask = compute_action_mask(frame_obs, agent_id)

            # Skip transitions where the taken action is masked (safety guard)
            if not action_mask[action_taken]:
                continue

            all_spatial.append(spatial)
            all_aux.append(aux)
            all_actions.append(action_taken)
            all_masks.append(action_mask)
            winner_demos += 1

        total_demos += winner_demos

    if total_demos == 0:
        print("WARNING: No demonstrations collected. Check that TacticalRuleAgent is playable.")
        return

    print(f"Collected {total_demos:,} transitions from {n_games} games.")

    spatial_arr = np.stack(all_spatial, axis=0).astype(np.float32)   # (N,15,13,13)
    aux_arr = np.stack(all_aux, axis=0).astype(np.float32)           # (N,7)
    actions_arr = np.array(all_actions, dtype=np.int64)              # (N,)
    masks_arr = np.stack(all_masks, axis=0).astype(bool)             # (N,6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        spatial=spatial_arr,
        aux=aux_arr,
        actions=actions_arr,
        action_masks=masks_arr,
    )
    print(f"Dataset saved to {output_path}  ({total_demos:,} samples, shape {spatial_arr.shape})")


# ===========================================================================
# Dataset — supports .npz only (spatial + aux + actions arrays)
# ===========================================================================

class TacticalBCDataset(Dataset):
    """
    In-memory dataset loaded from a .npz file produced by generate_dataset().

    Keys expected: "spatial" (N,15,13,13) float32, "aux" (N,7) float32,
                   "actions" (N,) int64, "action_masks" (N,6) bool.
    """

    def __init__(self, npz_path: Path) -> None:
        data = np.load(str(npz_path), allow_pickle=False)
        self._spatial: np.ndarray = data["spatial"].astype(np.float32)
        self._aux: np.ndarray = data["aux"].astype(np.float32)
        self._actions: np.ndarray = data["actions"].astype(np.int64)
        self._masks: np.ndarray = data["action_masks"].astype(bool)
        self._n: int = len(self._actions)
        print(f"Loaded {self._n:,} transitions from {npz_path}")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._spatial[idx].copy()),
            torch.from_numpy(self._aux[idx].copy()),
            torch.tensor(int(self._actions[idx]), dtype=torch.long),
            torch.from_numpy(self._masks[idx].copy()),
        )


# ===========================================================================
# Focal Loss (handles class imbalance; PLACE_BOMB ~15% of actions)
# ===========================================================================

def masked_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss with action masking.

    Invalid action logits are filled with -1e9 before log-softmax so the policy
    never learns to predict forbidden actions.  Samples whose target action is
    itself masked are excluded from the loss (prevents catastrophic gradient spikes).

    Args:
        logits:  (B, 6) raw action logits
        targets: (B,)   ground-truth action indices
        masks:   (B, 6) bool mask — True means action is valid
        gamma:   focal loss focusing parameter

    Returns:
        Scalar loss tensor.
    """
    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    focal_weight = (1.0 - pt) ** gamma
    loss = -focal_weight * log_pt

    # Guard: exclude samples where the target action is masked
    target_valid = masks.gather(1, targets.unsqueeze(1)).squeeze(1)
    valid_n = target_valid.float().sum().clamp(min=1.0)
    return (loss * target_valid.float()).sum() / valid_n


# ===========================================================================
# Training loop
# ===========================================================================

def train_tactical_bc(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 3e-4,
    val_split: float = 0.1,
    gamma_focal: float = 2.0,
    device: str = "auto",
    save_every: int = 5,
    init_from: Optional[Path] = None,
) -> Path:
    """
    Train BomberPolicyNet via Behavioral Cloning on TacticalRuleAgent rollouts.

    Training structure:
      - 90/10 train/val split with fixed seed 42
      - Adam + CosineAnnealingLR over all epochs
      - masked focal loss (gamma=2) per batch
      - early stopping with patience=5 (no val_loss improvement)
      - save every save_every epochs + save on new best val_loss

    Args:
        dataset_path: .npz file produced by generate_dataset()
        output_dir:   directory for checkpoint files
        epochs:       maximum training epochs
        batch_size:   mini-batch size
        lr:           Adam learning rate
        val_split:    fraction held out for validation
        gamma_focal:  focal loss gamma
        device:       "cuda" / "cpu" / "auto"
        save_every:   save checkpoint every N epochs
        init_from:    optional warm-start checkpoint (.pt)

    Returns:
        Path to best checkpoint.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Device: {dev}")

    # ── Dataset ──────────────────────────────────────────────────────── #
    full_dataset = TacticalBCDataset(dataset_path)
    n_total = len(full_dataset)

    val_n = max(1, int(n_total * val_split))
    train_n = n_total - val_n
    train_ds, val_ds = random_split(
        full_dataset,
        [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )

    num_workers = min(4, 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, num_workers),
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # ── Model ────────────────────────────────────────────────────────── #
    model = BomberPolicyNet().to(dev)
    if init_from is not None and Path(init_from).exists():
        print(f"Warm-start from {init_from}")
        model.init_from_bc(str(init_from), device=device)
    else:
        if init_from is not None:
            print(f"WARNING: init checkpoint not found: {init_from} — training from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt: Path = output_dir / "tactical_bc_best.pt"

    # ── Training loop ────────────────────────────────────────────────── #
    best_val_loss = float("inf")
    early_stop_patience = 5
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        n_processed = 0

        for sp, aux, act, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            sp = sp.to(dev)
            aux = aux.to(dev)
            act = act.to(dev)
            mask = mask.to(dev)

            logits = model.get_action_logits(sp, aux)
            loss = masked_focal_loss(logits, act, mask, gamma=gamma_focal)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_n = len(act)
            train_loss += loss.item() * batch_n
            train_correct += (logits.argmax(dim=1) == act).sum().item()
            n_processed += batch_n

        scheduler.step()
        train_loss /= max(n_processed, 1)
        train_acc = train_correct / max(n_processed, 1)

        # ── Validation ───────────────────────────────────────────────── #
        model.eval()
        val_loss = 0.0
        val_correct = 0
        n_val_processed = 0

        with torch.no_grad():
            for sp, aux, act, mask in val_loader:
                sp, aux, act, mask = sp.to(dev), aux.to(dev), act.to(dev), mask.to(dev)
                logits = model.get_action_logits(sp, aux)
                loss = masked_focal_loss(logits, act, mask, gamma=gamma_focal)
                batch_n = len(act)
                val_loss += loss.item() * batch_n
                val_correct += (logits.argmax(dim=1) == act).sum().item()
                n_val_processed += batch_n

        val_loss /= max(n_val_processed, 1)
        val_acc = val_correct / max(n_val_processed, 1)

        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Periodic checkpoint ───────────────────────────────────────── #
        if epoch % save_every == 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            ckpt_path = output_dir / f"tactical_bc_{epoch}ep_{ts}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(f"  -> Saved {ckpt_path.name}")

        # ── Best checkpoint + early stopping ──────────────────────────── #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_ckpt,
            )
            print(f"  -> New best: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {early_stop_patience} epochs)"
                )
                break

    print(f"\nBest val loss: {best_val_loss:.4f} -> {best_ckpt}")
    return best_ckpt


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tactical BC — generate TacticalRuleAgent rollouts and/or train BomberPolicyNet"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run self-rollout games and save winner demonstrations to dataset",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train BomberPolicyNet on the dataset with focal loss",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=200,
        metavar="N",
        help="Number of games to generate (default: 200)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        metavar="N",
        help="Mini-batch size (default: 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Adam learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        metavar="GAMMA",
        help="Focal loss gamma parameter (default: 2.0)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        metavar="N",
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional warm-start checkpoint (.pt) for training",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "data",
        metavar="DIR",
        help="Root data directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "checkpoints",
        metavar="DIR",
        help="Checkpoint output directory (default: <repo>/checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Base RNG seed for game generation (default: time-based)",
    )
    return parser


def _cli() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.generate and not args.train:
        parser.error("At least one of --generate or --train must be specified.")

    dataset_path: Path = args.data_dir / "tactical_bc_dataset.npz"

    if args.generate:
        print(f"=== Generating {args.n_games} games ===")
        generate_dataset(
            n_games=args.n_games,
            output_path=dataset_path,
            seed=args.seed,
        )

    if args.train:
        if not dataset_path.exists():
            parser.error(
                f"Dataset not found: {dataset_path}\n"
                "Run with --generate first, or point --data-dir to the correct directory."
            )
        print(f"\n=== Training on {dataset_path} ===")
        best_ckpt = train_tactical_bc(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gamma_focal=args.gamma,
            device=args.device,
            save_every=args.save_every,
            init_from=args.checkpoint,
        )
        print(f"\nTraining complete. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    _cli()
