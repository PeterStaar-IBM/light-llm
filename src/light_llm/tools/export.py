"""
Export a trained checkpoint to various inference formats.

Formats
-------
  onnx        – ONNX graph with dynamic batch/sequence axes
  torchscript – TorchScript traced module (.pt)
  safetensors – weights-only re-export

Examples
--------
llm-export --checkpoint checkpoints/best --format onnx --output exports/model.onnx
llm-export --checkpoint checkpoints/best --format torchscript --output exports/model.pt
llm-export --checkpoint checkpoints/best --format safetensors --output exports/weights.safetensors
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
import torch.nn as nn
import typer
from rich.console import Console
from safetensors.torch import load_file, save_file

from light_llm.models.transformer.model import Transformer
from light_llm.training.config import TrainingConfig

console = Console()
app = typer.Typer(help="Export a checkpoint to ONNX, TorchScript, or SafeTensors.")


@app.command()
def main(
    checkpoint: Annotated[str, typer.Option("--checkpoint", "-c", help="Checkpoint tag or .safetensors path.")],
    format:     Annotated[str, typer.Option("--format", "-f", help="Export format: onnx | torchscript | safetensors.")] = "onnx",
    output:     Annotated[str, typer.Option("--output", "-o", help="Output file path.")] = "exports/model",
    seq_len:    Annotated[int, typer.Option(help="[onnx/torchscript] Sequence length for tracing.")] = 128,
    batch_size: Annotated[int, typer.Option(help="[onnx/torchscript] Batch size for tracing.")] = 1,
    dtype:      Annotated[str, typer.Option(help="Export dtype: float32 | float16 | bfloat16.")] = "float32",
    opset:      Annotated[int, typer.Option(help="[onnx] ONNX opset version.")] = 17,
    verify:     Annotated[bool, typer.Option(help="[onnx] Verify outputs against PyTorch.")] = True,
) -> None:

    model = _load_model(checkpoint, dtype)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    match format:
        case "onnx":
            _export_onnx(model, out, seq_len, batch_size, opset, verify)
        case "torchscript":
            _export_torchscript(model, out, seq_len, batch_size)
        case "safetensors":
            _export_safetensors(model, out)
        case _:
            console.print(f"[red]Unknown format: {format!r}. Use onnx | torchscript | safetensors[/red]")
            raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_model(checkpoint: str, dtype: str) -> Transformer:
    ckpt = Path(checkpoint)
    weights_path = ckpt if ckpt.suffix == ".safetensors" else ckpt.with_suffix(".safetensors")
    cfg_path = Path(str(weights_path).replace(".safetensors", ".config.json"))

    if not cfg_path.exists():
        console.print(f"[red]Config not found: {cfg_path}[/red]")
        raise typer.Exit(1)

    train_cfg = TrainingConfig.model_validate_json(cfg_path.read_text())
    torch_dtype = _resolve_dtype(dtype)

    console.print(f"[cyan]Loading model…[/cyan]")
    model = Transformer(train_cfg.model)
    model.load_state_dict(load_file(str(weights_path)))
    model = model.to(dtype=torch_dtype).eval()
    console.print(f"[green]{model}[/green]\n")
    return model


# ---------------------------------------------------------------------------
# Format exporters
# ---------------------------------------------------------------------------

def _export_onnx(
    model: Transformer,
    out: Path,
    seq_len: int,
    batch_size: int,
    opset: int,
    verify: bool,
) -> None:
    import onnx

    if out.suffix != ".onnx":
        out = out.with_suffix(".onnx")

    dummy = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len))
    console.print(f"[cyan]Tracing with input {list(dummy.shape)}…[/cyan]")

    # Wrapper strips KV-cache from the public interface for static export
    class _StaticWrapper(nn.Module):
        def __init__(self, m: Transformer) -> None:
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.m(x)  # type: ignore[return-value]

    torch.onnx.export(
        _StaticWrapper(model),
        (dummy,),
        str(out),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits":    {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    onnx.checker.check_model(onnx.load(str(out)))
    console.print(f"[green]ONNX model saved → {out}[/green]")

    if verify:
        _verify_onnx(model, dummy, out)


def _verify_onnx(model: Transformer, dummy: torch.Tensor, out: Path) -> None:
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out))
        with torch.no_grad():
            pt_out = model(dummy).float().numpy()
        ort_out = sess.run(["logits"], {"input_ids": dummy.numpy()})[0]
        max_diff = float(np.abs(pt_out - ort_out).max())
        console.print(f"  [dim]Max output diff (PT vs ORT): {max_diff:.6f}[/dim]")
        symbol = "[green]✓[/green]" if max_diff < 1e-3 else "[yellow]⚠[/yellow]"
        console.print(f"  {symbol} Verification {'passed' if max_diff < 1e-3 else 'large diff — check dtype/opset'}")
    except ImportError:
        console.print("  [dim]onnxruntime not installed; skipping verification.[/dim]")


def _export_torchscript(model: Transformer, out: Path, seq_len: int, batch_size: int) -> None:
    if out.suffix != ".pt":
        out = out.with_suffix(".pt")

    dummy = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len))
    console.print("[cyan]Tracing TorchScript…[/cyan]")
    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy,))
    traced.save(str(out))
    console.print(f"[green]TorchScript model saved → {out}[/green]")


def _export_safetensors(model: Transformer, out: Path) -> None:
    if out.suffix != ".safetensors":
        out = out.with_suffix(".safetensors")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out))
    console.print(f"[green]SafeTensors weights saved → {out}[/green]")


def _resolve_dtype(dtype: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.float32)
