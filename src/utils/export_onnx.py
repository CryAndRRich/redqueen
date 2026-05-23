"""
ONNX export pipeline for BomberPolicyNet.

Exports the policy head (logits only — no value) with fixed input shapes
so onnxruntime can run inference without dynamic axes.

Input shapes (fixed):
  spatial: (1, 15, 13, 13) float32
  aux:     (1, 7)          float32

Output:
  logits:  (1, 6)          float32  — raw action logits

Usage:
    python -m src.utils.export_onnx --checkpoint checkpoints/bc_best.pt
    python -m src.utils.export_onnx --checkpoint checkpoints/ppo_best.pt --output exports/model.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.policy_network import BomberPolicyNet  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────── #
# ONNX-compatible policy wrapper                                               #
# ─────────────────────────────────────────────────────────────────────────── #

class _PolicyONNXWrapper(torch.nn.Module):
    """Wraps BomberPolicyNet to export only the logit output."""

    def __init__(self, net: BomberPolicyNet) -> None:
        super().__init__()
        self.net = net

    def forward(
        self,
        spatial: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.net(spatial, aux)
        return logits


# ─────────────────────────────────────────────────────────────────────────── #
# Export function                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    verify: bool = True,
) -> Path:
    """
    Load BomberPolicyNet from checkpoint and export to ONNX.

    Args:
        checkpoint_path: .pt file with model_state_dict
        output_path:     destination .onnx file
        opset:           ONNX opset version (17 recommended)
        verify:          run a test inference to confirm output matches PyTorch

    Returns:
        Path to the exported ONNX file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    net = BomberPolicyNet.load(str(checkpoint_path), device="cpu")
    net.eval()
    wrapper = _PolicyONNXWrapper(net)
    wrapper.eval()

    # Dummy inputs (fixed batch size = 1)
    dummy_spatial = torch.zeros(1, 15, 13, 13, dtype=torch.float32)
    dummy_aux     = torch.zeros(1, 7,       dtype=torch.float32)

    # Export
    print(f"Exporting to {output_path} (opset {opset}) …")
    torch.onnx.export(
        wrapper,
        (dummy_spatial, dummy_aux),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["spatial", "aux"],
        output_names=["logits"],
        dynamic_axes=None,  # fixed shapes for best onnxruntime performance
    )
    print(f"Exported: {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

    # ── Verify ────────────────────────────────────────────────────────── #
    if verify:
        _verify_onnx(wrapper, output_path, dummy_spatial, dummy_aux)

    return output_path


def _verify_onnx(
    wrapper: _PolicyONNXWrapper,
    onnx_path: Path,
    dummy_spatial: torch.Tensor,
    dummy_aux: torch.Tensor,
) -> None:
    """Run PyTorch and ONNX inference and assert outputs are close."""
    try:
        import onnxruntime as ort
        import onnx
    except ImportError:
        print("onnxruntime / onnx not installed — skipping verification.")
        return

    # Check ONNX model validity
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    print("ONNX model check: OK")

    # Compare outputs
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    spatial_np = dummy_spatial.numpy()
    aux_np     = dummy_aux.numpy()

    ort_outputs = sess.run(None, {"spatial": spatial_np, "aux": aux_np})
    ort_logits  = ort_outputs[0]

    with torch.no_grad():
        pt_logits = wrapper(dummy_spatial, dummy_aux).numpy()

    max_diff = float(np.abs(ort_logits - pt_logits).max())
    print(f"Max abs diff (PyTorch vs ONNX): {max_diff:.6f}")
    assert max_diff < 1e-4, f"ONNX output deviates too much: {max_diff}"
    print("Verification PASSED ✓")

    # Benchmark inference speed
    _benchmark(sess, spatial_np, aux_np)


def _benchmark(
    sess,
    spatial_np: np.ndarray,
    aux_np: np.ndarray,
    n_runs: int = 1000,
) -> None:
    """Print median inference time per step."""
    import time
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"spatial": spatial_np, "aux": aux_np})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median_ms = times[len(times) // 2]
    p95_ms    = times[int(len(times) * 0.95)]
    print(f"Inference speed ({n_runs} runs):  median {median_ms:.2f} ms  p95 {p95_ms:.2f} ms")
    if p95_ms > 100:
        print("WARNING: p95 > 100 ms — may exceed competition inference budget!")
    else:
        print("Inference budget check: OK (< 100 ms)")


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Export BomberPolicyNet to ONNX")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Output .onnx path (default: exports/<checkpoint_name>.onnx)",
    )
    parser.add_argument("--opset",     type=int,  default=17)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = _ROOT / "exports" / (args.checkpoint.stem + ".onnx")

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    _cli()
