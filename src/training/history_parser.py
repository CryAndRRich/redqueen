"""
Phase 0 — BC data mining from history_game/ match files.

Extracts (spatial, aux, action, mask) tuples from real competition matches.
Quality filter: rank=0 winner AND survival ≥ 120 steps AND ≥ 5 bombs placed.

Output: a directory containing 4 memmap .npy files + metadata:
  spatial.npy       (N, 15, 13, 13) float32  — written via np.memmap
  aux.npy           (N, 7)          float32
  actions.npy       (N,)            int64
  action_masks.npy  (N, 6)          bool
  _n.npy            scalar          int64     — actual number of valid rows

Two-pass approach avoids accumulating data in RAM:
  Pass 1 — scan metadata only, count N_total (no feature extraction)
  Pass 2 — extract features, write directly to pre-allocated memmaps

Frame alignment:
  obs_t  = history[t]           (state after step t's actions were applied)
  label  = history[t+1].actions[agent_id]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.feature_extractor import extract_features, count_boxes  # noqa: E402
from src.logic.action_masking import compute_action_mask               # noqa: E402

# ─────────────────────────────────────────────────────────────────────────── #
# Quality filter                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

MIN_SURVIVAL: int = 120
MIN_BOMBS_PLACED: int = 5


def _count_bombs_placed(history: list[dict], agent_idx: int) -> int:
    return sum(
        1
        for frame in history[1:]
        if frame["actions"] is not None and frame["actions"][agent_idx] == 5
    )


def _is_quality(match: dict, agent_idx: int) -> bool:
    """Return True if the agent's trajectory is worth learning from."""
    return (
        match["ranks"][agent_idx] == 0
        and match["survival_steps"][agent_idx] >= MIN_SURVIVAL
        and _count_bombs_placed(match["history"], agent_idx) >= MIN_BOMBS_PLACED
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Frame reconstruction                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def _frame_to_obs(frame: dict) -> dict:
    """Convert a history frame to the engine obs dict format."""
    return {
        "map":     np.array(frame["map"], dtype=np.int8),
        "players": np.array(frame["players"], dtype=np.int8),
        "bombs":   np.array(frame["bombs"],   dtype=np.int8) if frame["bombs"] else np.zeros((0, 4), dtype=np.int8),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Per-file extraction                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def extract_from_match(
    match: dict,
    agent_idx: int,
    total_steps: int = 500,
) -> Generator[tuple[np.ndarray, np.ndarray, int, np.ndarray], None, None]:
    """
    Yield (spatial, aux, action, action_mask) for each quality step.

    Args:
        match:      parsed JSON dict
        agent_idx:  player index to extract demonstrations for
        total_steps: max game steps (for step_normalized)
    """
    history = match["history"]
    initial_boxes = count_boxes(np.array(history[0]["map"], dtype=np.int8))
    kills = 0
    boxes_destroyed = 0

    for t in range(len(history) - 1):
        frame_t = history[t]
        frame_t1 = history[t + 1]

        if frame_t1["actions"] is None:
            continue

        if not frame_t["alive"][agent_idx]:
            break

        action = int(frame_t1["actions"][agent_idx])

        obs_t = _frame_to_obs(frame_t)

        players_t  = np.array(frame_t["players"])
        players_t1 = np.array(frame_t1["players"])
        prev_enemies = sum(int(players_t[i][2])  for i in range(4) if i != agent_idx)
        curr_enemies = sum(int(players_t1[i][2]) for i in range(4) if i != agent_idx)
        kills += max(0, prev_enemies - curr_enemies)

        grid_t  = np.array(frame_t["map"],   dtype=np.int8)
        grid_t1 = np.array(frame_t1["map"],  dtype=np.int8)
        boxes_destroyed += int(((grid_t == 2) & (grid_t1 != 2)).sum())

        boxes_now = int((grid_t == 2).sum())

        spatial, aux = extract_features(
            obs=obs_t,
            agent_id=agent_idx,
            step=frame_t["step"],
            total_steps=total_steps,
            initial_boxes=max(1, initial_boxes),
            boxes_remaining=boxes_now,
            my_kills=kills,
            my_boxes_destroyed=boxes_destroyed,
        )
        mask = compute_action_mask(obs_t, agent_idx)

        yield spatial, aux, action, mask


# ─────────────────────────────────────────────────────────────────────────── #
# Pass 1 — fast count (no feature extraction)                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def _pass1_count(
    json_files: list[Path],
    min_survival: int,
    min_bombs: int,
) -> tuple[int, list[tuple[Path, int]]]:
    """
    Scan all files, apply quality filter on metadata only, count N_total.

    Returns:
        n_total:      total valid transitions across all quality agents
        quality_list: [(file_path, agent_idx), ...] — agents that pass filter
    """
    n_total = 0
    quality_list: list[tuple[Path, int]] = []

    for fp in tqdm(json_files, desc="Pass 1/2 — counting transitions"):
        try:
            with fp.open() as f:
                match = json.load(f)
        except Exception:
            continue

        n_frames = len(match["history"])
        for agent_idx in range(4):
            if (
                match["ranks"][agent_idx] != 0
                or match["survival_steps"][agent_idx] < min_survival
                or _count_bombs_placed(match["history"], agent_idx) < min_bombs
            ):
                continue

            # Count alive frames without feature extraction
            n_alive = 0
            for t in range(n_frames - 1):
                if not match["history"][t]["alive"][agent_idx]:
                    break
                if match["history"][t + 1]["actions"] is not None:
                    n_alive += 1

            if n_alive > 0:
                n_total += n_alive
                quality_list.append((fp, agent_idx))

        del match  # explicit GC after each file

    return n_total, quality_list


# ─────────────────────────────────────────────────────────────────────────── #
# Main parser                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def parse_history(
    history_dir: Path,
    output_path: Path,
    max_files: int | None = None,
    min_survival: int = MIN_SURVIVAL,
    min_bombs: int = MIN_BOMBS_PLACED,
) -> None:
    """
    Scan history_dir for JSON match files, extract quality demos, save to output_path/.

    Two-pass approach: Pass 1 counts transitions (fast, no feature extraction),
    Pass 2 extracts features and writes directly to pre-allocated memmaps on disk.
    Peak RAM = O(one match file + one batch of features) regardless of dataset size.

    Args:
        history_dir:  path to history_game/ root (recursively searched)
        output_path:  destination directory (will contain .npy memmap files)
        max_files:    limit number of JSON files processed (None = all)
        min_survival: override quality filter survival threshold
        min_bombs:    override quality filter bombs threshold
    """
    json_files = sorted(history_dir.rglob("*.json"))
    if max_files:
        json_files = json_files[:max_files]

    print(f"Found {len(json_files):,} JSON files in {history_dir}")

    # ── Pass 1: count quality transitions ──────────────────────────────── #
    n_total, quality_list = _pass1_count(json_files, min_survival, min_bombs)

    print(
        f"Pass 1 complete: {len(quality_list)} quality trajectories, "
        f"{n_total:,} transitions"
    )

    if n_total == 0:
        print("No quality transitions found. Try relaxing the quality filter.")
        return

    # ── Pre-allocate memmaps ───────────────────────────────────────────── #
    output_path.mkdir(parents=True, exist_ok=True)

    sp_mm   = np.memmap(output_path / "spatial.npy",      dtype="float32", mode="w+", shape=(n_total, 15, 13, 13))
    aux_mm  = np.memmap(output_path / "aux.npy",          dtype="float32", mode="w+", shape=(n_total, 7))
    act_mm  = np.memmap(output_path / "actions.npy",      dtype="int64",   mode="w+", shape=(n_total,))
    mask_mm = np.memmap(output_path / "action_masks.npy", dtype="bool",    mode="w+", shape=(n_total, 6))

    disk_gb = n_total * (15 * 13 * 13 * 4 + 7 * 4 + 8 + 6) / 1e9
    print(f"Pre-allocated {disk_gb:.2f} GB on disk at {output_path}/")

    # ── Pass 2: extract features, fill memmaps ─────────────────────────── #
    idx = 0
    skipped = 0

    for fp, agent_idx in tqdm(quality_list, desc="Pass 2/2 — extracting features"):
        try:
            with fp.open() as f:
                match = json.load(f)
        except Exception:
            skipped += 1
            continue

        for spatial, aux, action, mask in extract_from_match(match, agent_idx):
            if idx >= n_total:
                break  # guard against count mismatch
            sp_mm[idx]   = spatial
            aux_mm[idx]  = aux
            act_mm[idx]  = action
            mask_mm[idx] = mask
            idx += 1

        del match

    actual_n = idx

    # Flush memmaps to disk before reading back
    del sp_mm, aux_mm, act_mm, mask_mm

    # Save actual count (may differ slightly from estimate due to off-by-one edge cases)
    np.save(output_path / "_n.npy", np.array(actual_n, dtype=np.int64))

    if actual_n != n_total:
        print(f"  Note: actual transitions ({actual_n:,}) ≠ estimated ({n_total:,})")

    print(f"\nSaved dataset → {output_path}/  ({actual_n:,} samples)")
    print(f"  Files: spatial.npy, aux.npy, actions.npy, action_masks.npy, _n.npy")
    print(f"  Skipped {skipped} files (parse errors)")

    # ── Action distribution ────────────────────────────────────────────── #
    actions_final = np.memmap(output_path / "actions.npy", dtype="int64", mode="r", shape=(actual_n,))
    from collections import Counter
    dist = Counter(int(a) for a in actions_final)
    names = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]
    print("Action distribution:")
    for a in range(6):
        pct = 100.0 * dist.get(a, 0) / max(actual_n, 1)
        print(f"  {names[a]:5s} ({a}): {dist.get(a, 0):7d}  {pct:.1f}%")
    del actions_final


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Extract BC dataset from history_game/")
    parser.add_argument(
        "--history-dir", type=Path,
        default=_ROOT / "history_game",
        help="Path to history_game/ directory",
    )
    parser.add_argument(
        "--output", type=Path,
        default=_ROOT / "data" / "bc_dataset",
        help="Output directory (will contain .npy memmap files)",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--min-survival", type=int, default=MIN_SURVIVAL)
    parser.add_argument("--min-bombs",    type=int, default=MIN_BOMBS_PLACED)
    args = parser.parse_args()

    parse_history(
        args.history_dir,
        args.output,
        max_files=args.max_files,
        min_survival=args.min_survival,
        min_bombs=args.min_bombs,
    )


if __name__ == "__main__":
    _cli()
