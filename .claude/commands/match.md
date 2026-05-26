# /match — Run a local headless test match

Run a headless test match between our agent and rule-based opponents.

## Command

```bash
python scripts/run_local_match.py --headless --agent agent/agent.py
```

Add `--visual` for live pygame visualization (requires display):

```bash
python scripts/run_local_match.py --visual --agent agent/agent.py
```

## Prerequisites

ONNX model must be exported first. Run `/export` if `agent/model.onnx` is missing.

## Output

Prints final ranks, kills, boxes destroyed, and steps survived per player.
