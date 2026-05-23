"""
Phase 0 — BC data mining from history_game/ match files.

Extracts (spatial, aux, action, mask) tuples from real competition matches.
Quality filter: rank=0 winner AND survival ≥ 120 steps AND ≥ 5 bombs placed.

Output: data/bc_dataset.npz
  spatial:      (N, 15, 13, 13) float32
  aux:          (N, 7)          float32
  actions:      (N,)            int64
  action_masks: (N, 6)          bool

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

        # Skip if next frame has no actions (shouldn't happen, but guard)
        if frame_t1["actions"] is None:
            continue

        # Skip if our agent is dead at frame t
        if not frame_t["alive"][agent_idx]:
            break

        action = int(frame_t1["actions"][agent_idx])

        obs_t = _frame_to_obs(frame_t)

        # Update external trackers
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
    Scan history_dir for JSON match files, extract quality demos, save .npz.

    Args:
        history_dir:  path to history_game/ root (recursively searched)
        output_path:  destination .npz file
        max_files:    limit number of JSON files processed (None = all)
        min_survival: override quality filter survival threshold
        min_bombs:    override quality filter bombs threshold
    """
    json_files = sorted(history_dir.rglob("*.json"))
    if max_files:
        json_files = json_files[:max_files]

    spatials: list[np.ndarray] = []
    auxes:    list[np.ndarray] = []
    actions_: list[int]        = []
    masks_:   list[np.ndarray] = []

    skipped = 0
    parsed = 0

    for fp in tqdm(json_files, desc="Parsing matches"):
        try:
            with fp.open() as f:
                match = json.load(f)
        except Exception:
            skipped += 1
            continue

        for agent_idx in range(4):
            # Override thresholds if requested
            if (
                match["ranks"][agent_idx] != 0
                or match["survival_steps"][agent_idx] < min_survival
                or _count_bombs_placed(match["history"], agent_idx) < min_bombs
            ):
                continue

            for spatial, aux, action, mask in extract_from_match(match, agent_idx):
                spatials.append(spatial)
                auxes.append(aux)
                actions_.append(action)
                masks_.append(mask)

        parsed += 1

    print(f"Parsed {parsed} files, skipped {skipped}, extracted {len(actions_)} transitions")

    if not actions_:
        print("No quality transitions found. Try relaxing the quality filter.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        spatial=np.stack(spatials).astype(np.float32),
        aux=np.stack(auxes).astype(np.float32),
        actions=np.array(actions_, dtype=np.int64),
        action_masks=np.stack(masks_).astype(np.bool_),
    )
    print(f"Saved dataset → {output_path}  ({len(actions_)} samples)")

    # Class distribution
    from collections import Counter
    dist = Counter(actions_)
    total = len(actions_)
    print("Action distribution:")
    names = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]
    for a in range(6):
        pct = 100.0 * dist.get(a, 0) / total
        print(f"  {names[a]:5s} ({a}): {dist.get(a, 0):7d}  {pct:.1f}%")


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
        default=_ROOT / "data" / "bc_dataset.npz",
        help="Output .npz path",
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
