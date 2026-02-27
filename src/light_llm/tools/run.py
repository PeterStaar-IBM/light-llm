"""
Run inference (text generation) from a trained checkpoint.

Examples
--------
llm-run \\
    --checkpoint checkpoints/best \\
    --tokenizer  gpt2 \\
    --prompt     "Once upon a time" \\
    --max-tokens 200 --temperature 0.8 --top-p 0.9

# Interactive REPL
llm-run --checkpoint checkpoints/best --tokenizer gpt2 --interactive
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from rich.console import Console
from safetensors.torch import load_file

from light_llm.models.transformer.model import Transformer
from light_llm.tokenizer import HFTokenizer
from light_llm.training.config import TrainingConfig

console = Console()
app = typer.Typer(help="Generate text from a trained checkpoint.")


@app.command()
def main(
    checkpoint:  Annotated[str, typer.Option("--checkpoint", "-c", help="Checkpoint tag or .safetensors path.")],
    tokenizer:   Annotated[str, typer.Option("--tokenizer", "-t", help="HF model name or saved tokenizer path.")],
    prompt:      Annotated[Optional[str], typer.Option("--prompt", "-p", help="Prompt text.")] = None,
    max_tokens:  Annotated[int,   typer.Option(help="Maximum tokens to generate.")] = 200,
    temperature: Annotated[float, typer.Option(help="Sampling temperature (0 = greedy).")] = 1.0,
    top_k:       Annotated[Optional[int],   typer.Option(help="Top-k sampling.")] = None,
    top_p:       Annotated[Optional[float], typer.Option(help="Nucleus (top-p) sampling.")] = None,
    repetition_penalty: Annotated[float, typer.Option(help="Repetition penalty.")] = 1.0,
    interactive: Annotated[bool, typer.Option("--interactive", "-I", help="Interactive generation loop.")] = False,
    dtype:       Annotated[str, typer.Option(help="Inference dtype: float32 | float16 | bfloat16.")] = "bfloat16",
    device:      Annotated[str, typer.Option(help="Device: cuda | cpu.")] = "cuda",
) -> None:

    console.print(f"[cyan]Loading tokenizer: {tokenizer}[/cyan]")
    tok = HFTokenizer.from_pretrained(tokenizer)

    model, device_ = _load_model(checkpoint, dtype, device)

    eos_id = tok.eos_token_id
    torch_dtype = _resolve_dtype(dtype)

    def generate(text: str) -> str:
        ids = tok.encode(text, add_special_tokens=True)
        prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device_)
        with torch.inference_mode(), torch.amp.autocast(str(device_), dtype=torch_dtype):
            out = model.generate(
                prompt_tensor,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-8),
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_id,
            )
        return tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)

    if interactive:
        console.print("[bold]Interactive mode — Ctrl-C to exit[/bold]\n")
        while True:
            try:
                text = console.input("[bold cyan]Prompt>[/bold cyan] ").strip()
                if not text:
                    continue
                result = generate(text)
                console.print(f"\n[bold]{text}[/bold]{result}\n")
            except KeyboardInterrupt:
                console.print("\nBye!")
                break
    else:
        if not prompt:
            console.print("[red]--prompt is required in non-interactive mode[/red]")
            raise typer.Exit(1)
        result = generate(prompt)
        console.print(f"\n[bold]{prompt}[/bold]{result}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint: str, dtype: str, device: str) -> tuple[Transformer, torch.device]:
    ckpt = Path(checkpoint)
    weights_path = ckpt if ckpt.suffix == ".safetensors" else ckpt.with_suffix(".safetensors")
    cfg_path = Path(str(weights_path).replace(".safetensors", ".config.json"))

    if not cfg_path.exists():
        console.print(f"[red]Config file not found: {cfg_path}[/red]")
        raise typer.Exit(1)

    train_cfg = TrainingConfig.model_validate_json(cfg_path.read_text())
    device_ = torch.device(device if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_dtype(dtype)

    console.print(f"[cyan]Loading model from {weights_path}[/cyan]")
    model = Transformer(train_cfg.model)
    model.load_state_dict(load_file(str(weights_path), device=str(device_)))
    model = model.to(device_, dtype=torch_dtype).eval()
    console.print(f"[green]{model}[/green]\n")
    return model, device_


def _resolve_dtype(dtype: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.float32)
