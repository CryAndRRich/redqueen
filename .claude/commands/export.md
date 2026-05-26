# /export — Export checkpoint to ONNX + TorchScript

Export a BomberPolicyNet checkpoint to `model.onnx` and `model.pt`.

## Command

```bash
python -m src.utils.export_onnx --checkpoint <path_to_checkpoint>
```

Example:

```bash
python -m src.utils.export_onnx --checkpoint checkpoints/ppo_curriculum_best_20260526_1200.pt
```

## Notes

- Checkpoint files are in `checkpoints/`, named `{phase}_{description}_{timestamp}.pt`
- Outputs go to `exports/model.onnx` and `exports/model.pt`
- After export, copy both files to `agent/` before submitting:
  ```bash
  cp exports/model.onnx agent/model.onnx
  cp exports/model.pt   agent/model.pt
  ```
- Validate inference latency is **< 100 ms on CPU** after export
