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

from src.models.policy_network import BomberPolicyNet


class _PolicyONNXWrapper(torch.nn.Module):
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


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    verify: bool = True,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    net = BomberPolicyNet.load(str(checkpoint_path), device="cpu")
    net.eval()
    wrapper = _PolicyONNXWrapper(net)
    wrapper.eval()

    dummy_spatial = torch.zeros(1, 15, 13, 13, dtype=torch.float32)
    dummy_aux     = torch.zeros(1, 7,       dtype=torch.float32)

    print(f"Exporting to {output_path} (opset {opset}) ...")
    torch.onnx.export(
        wrapper,
        (dummy_spatial, dummy_aux),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["spatial", "aux"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )
    print(f"Exported: {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

    if verify:
        _verify_onnx(wrapper, output_path, dummy_spatial, dummy_aux)

    return output_path


def _verify_onnx(
    wrapper: _PolicyONNXWrapper,
    onnx_path: Path,
    dummy_spatial: torch.Tensor,
    dummy_aux: torch.Tensor,
) -> None:
    try:
        import onnxruntime as ort
        import onnx
    except ImportError:
        print("onnxruntime / onnx not installed — skipping verification.")
        return

    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    print("ONNX model check: OK")

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
    print("Verification PASSED")

    _benchmark(sess, spatial_np, aux_np)


def _benchmark(
    sess,
    spatial_np: np.ndarray,
    aux_np: np.ndarray,
    n_runs: int = 1000,
) -> None:
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


def export_torchscript(
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint for TorchScript export: {checkpoint_path}")
    net = BomberPolicyNet.load(str(checkpoint_path), device="cpu")
    net.eval()
    wrapper = _PolicyONNXWrapper(net)
    wrapper.eval()

    dummy_spatial = torch.zeros(1, 15, 13, 13, dtype=torch.float32)
    dummy_aux     = torch.zeros(1, 7,       dtype=torch.float32)

    print(f"Tracing to TorchScript: {output_path} ...")
    traced = torch.jit.trace(wrapper, (dummy_spatial, dummy_aux))
    torch.jit.save(traced, str(output_path))
    print(f"TorchScript saved: {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

    loaded = torch.jit.load(str(output_path), map_location="cpu")
    with torch.no_grad():
        out = loaded(dummy_spatial, dummy_aux)
    assert out.shape == (1, 6), f"Unexpected output shape: {out.shape}"
    print("TorchScript verification PASSED")

    return output_path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Export BomberPolicyNet to ONNX and/or TorchScript")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Output .onnx path (default: exports/<checkpoint_name>.onnx)",
    )
    parser.add_argument("--opset",          type=int,  default=17)
    parser.add_argument("--no-verify",      action="store_true")
    parser.add_argument("--torchscript",    action="store_true",
                        help="Also export a TorchScript .pt file alongside the ONNX model")
    args = parser.parse_args()

    if args.output is None:
        args.output = _ROOT / "exports" / (args.checkpoint.stem + ".onnx")

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset=args.opset,
        verify=not args.no_verify,
    )

    if args.torchscript:
        ts_output = args.output.with_suffix(".pt")
        export_torchscript(
            checkpoint_path=args.checkpoint,
            output_path=ts_output,
        )


if __name__ == "__main__":
    _cli()
