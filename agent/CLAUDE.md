# agent/ — Submission Entry Point & Rule-Based Opponents

## Files

| File | Role |
|------|------|
| `agent.py` | **Submission entry point** — self-contained, ONNX Runtime inference, TorchScript fallback |
| `random_agent.py` | Curriculum Stage 0 opponent |
| `simple_rule_agent.py` | Curriculum Stage 1 opponent |
| `smarter_rule_agent.py` | Curriculum Stage 2/3 opponent |
| `tactical_rule_agent.py` | Curriculum Stage 4 opponent |
| `trapper_rule_agent.py` | Curriculum Stage 5 opponent |
| `genius_rule_agent.py` | Curriculum Stage 6 opponent + BC teacher |
| `__init__.py` | Exports all agents including TrapperRuleAgent |

## Key Rules for agent.py

- All feature extraction and action masking logic is **inlined** — no imports from `src/`
- **No PyTorch** — ONNX Runtime only for inference
- After training: copy `exports/model.onnx` and `exports/model.pt` here

## Submission Format

```bash
zip -j submission.zip agent/agent.py agent/model.onnx agent/model.pt
```

- **No `requirements.txt`** — causes `requirements_txt_forbidden` rejection
- **Validate**: `unzip -l submission.zip | grep agent.py` must show `agent.py` (not `agent/agent.py`)
