# /submit — Package submission.zip

Package `agent.py` + `model.onnx` + `model.pt` into a valid submission zip.

## Full sequence

```bash
# 1. Export checkpoint (if not done already)
python -m src.utils.export_onnx --checkpoint checkpoints/<best_ckpt>.pt

# 2. Copy exports to agent/
cp exports/model.onnx agent/model.onnx
cp exports/model.pt   agent/model.pt

# 3. Create zip (from project root — NOT from inside agent/)
zip -j submissions/submission_$(date +%Y%m%d_%H%M).zip agent/agent.py agent/model.onnx agent/model.pt

# 4. Validate — agent.py must appear at root, NOT as agent/agent.py
unzip -l submissions/submission_*.zip | grep agent.py
```

## Critical warnings

- **DO NOT include `requirements.txt`** — causes `requirements_txt_forbidden` error and immediate rejection
- **`agent.py` must be at the zip ROOT** — if it shows as `agent/agent.py` in step 4, the evaluation engine cannot find it and the score is ~0
- The `-j` flag in the `zip` command strips directory prefixes — do not remove it
