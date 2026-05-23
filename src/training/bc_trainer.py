"""
Phase 2 — Behavioral Cloning trainer.

Trains BomberPolicyNet on (spatial, aux) → action dataset with Focal Loss
to handle class imbalance (PLACE_BOMB is ~15% of actions).

Reads:  data/bc_dataset.npz  (produced by history_parser.py)
Writes: checkpoints/bc_{epoch}ep_{timestamp}.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.policy_network import BomberPolicyNet  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────── #
# Focal Loss                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Reduces the relative loss for well-classified examples,
    focuses on hard / rare samples (e.g. PLACE_BOMB action).

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        # log_probs[i, targets[i]]
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────── #
# Masked loss: only compute over valid (unmasked) actions                      #
# ─────────────────────────────────────────────────────────────────────────── #

def masked_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Apply action mask before computing focal loss.
    Fills invalid action logits with -1e9 before softmax.
    """
    # masks: (B, 6) bool — True = valid
    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    focal_weight = (1.0 - pt) ** gamma
    return (-focal_weight * log_pt).mean()


# ─────────────────────────────────────────────────────────────────────────── #
# Training loop                                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def train_bc(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 3e-4,
    val_split: float = 0.1,
    gamma_focal: float = 2.0,
    device: str = "auto",
    save_every: int = 10,
    init_from: Path | None = None,
) -> Path:
    """
    Train BomberPolicyNet via Behavioral Cloning.

    Args:
        dataset_path: .npz file from history_parser.py
        output_dir:   directory for checkpoints
        epochs:       number of training epochs
        batch_size:   mini-batch size
        lr:           Adam learning rate
        val_split:    fraction of data held out for validation
        gamma_focal:  focal loss gamma parameter
        device:       "cuda" / "cpu" / "auto"
        save_every:   save checkpoint every N epochs
        init_from:    optional .pt checkpoint to warm-start from

    Returns:
        Path to best checkpoint
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Device: {dev}")

    # ── Load dataset ─────────────────────────────────────────────────── #
    print(f"Loading dataset from {dataset_path} …")
    data = np.load(dataset_path)
    spatial_np = data["spatial"].astype(np.float32)      # (N, 15, 13, 13)
    aux_np     = data["aux"].astype(np.float32)           # (N, 7)
    actions_np = data["actions"].astype(np.int64)         # (N,)
    masks_np   = data["action_masks"].astype(bool)        # (N, 6)
    N = len(actions_np)
    print(f"Dataset size: {N:,} transitions")

    dataset = TensorDataset(
        torch.from_numpy(spatial_np),
        torch.from_numpy(aux_np),
        torch.from_numpy(actions_np),
        torch.from_numpy(masks_np),
    )
    val_n = max(1, int(N * val_split))
    train_n = N - val_n
    train_ds, val_ds = random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device == "cuda"))

    # ── Model ────────────────────────────────────────────────────────── #
    model = BomberPolicyNet().to(dev)
    if init_from and init_from.exists():
        print(f"Warm-start from {init_from}")
        model.init_from_bc(str(init_from), device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ─────────────────────────────────────────────────────── #
    best_val_loss = float("inf")
    best_ckpt: Path = output_dir / "bc_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0

        for sp, aux, act, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            sp   = sp.to(dev)
            aux  = aux.to(dev)
            act  = act.to(dev)
            mask = mask.to(dev)

            logits = model.get_action_logits(sp, aux)
            loss = masked_focal_loss(logits, act, mask, gamma=gamma_focal)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(act)
            correct += (logits.argmax(dim=1) == act).sum().item()

        scheduler.step()
        train_loss /= train_n
        train_acc   = correct / train_n

        # ── Validation ───────────────────────────────────────────────── #
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for sp, aux, act, mask in val_loader:
                sp, aux, act, mask = sp.to(dev), aux.to(dev), act.to(dev), mask.to(dev)
                logits = model.get_action_logits(sp, aux)
                loss = masked_focal_loss(logits, act, mask, gamma=gamma_focal)
                val_loss += loss.item() * len(act)
                val_correct += (logits.argmax(dim=1) == act).sum().item()

        val_loss /= val_n
        val_acc   = val_correct / val_n

        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Save checkpoint ───────────────────────────────────────────── #
        if epoch % save_every == 0:
            ts = time.strftime("%Y%m%d_%H%M%S")
            ckpt_path = output_dir / f"bc_{epoch}ep_{ts}.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "val_acc": val_acc}, ckpt_path)
            print(f"  → Saved {ckpt_path.name}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "val_acc": val_acc}, best_ckpt)

    print(f"\nBest val loss: {best_val_loss:.4f} → {best_ckpt}")
    return best_ckpt


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train BomberPolicyNet via Behavioral Cloning")
    parser.add_argument("--dataset",    type=Path, default=_ROOT / "data" / "bc_dataset.npz")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "checkpoints")
    parser.add_argument("--epochs",     type=int,  default=50)
    parser.add_argument("--batch-size", type=int,  default=512)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--gamma",      type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--save-every", type=int,  default=10)
    parser.add_argument("--init-from",  type=Path, default=None)
    parser.add_argument("--device",     type=str,  default="auto")
    args = parser.parse_args()

    train_bc(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma_focal=args.gamma,
        device=args.device,
        save_every=args.save_every,
        init_from=args.init_from,
    )


if __name__ == "__main__":
    _cli()
