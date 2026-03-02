# Installation Notes

## Standard install

```bash
uv sync
```

This installs the base package and all core dependencies. Optional extras:

- `dev` — pytest, ruff, mypy
- `analysis` — networkx, scipy, matplotlib
- `flash` — flash-attn (see below)

## Installing with flash-attn (`--all-extras`)

`flash-attn` has no prebuilt wheels for `aarch64` (e.g. DGX Spark) and must be compiled from
source. This requires:

1. **`python3.12-dev`** — provides `Python.h` needed by the CUDA extension build:
   ```bash
   sudo apt-get install -y python3.12-dev
   ```

2. **PyTorch from the `cu130` index** — `pyproject.toml` is already configured for this. The
   system CUDA toolkit version must match what PyTorch was compiled against; this project targets
   CUDA 13.0.

3. **Controlled build** — compiling 73 CUDA kernels for all architectures in parallel exhausts
   memory (OOM-killer kills `nvcc`). Limit parallelism and target only the architectures you
   actually have:

   ```bash
   MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="12.0;12.1" uv sync --all-extras
   ```

   | Variable | Purpose | Value |
   |---|---|---|
   | `MAX_JOBS` | Ninja parallel job limit | 4 (lower if still OOM) |
   | `TORCH_CUDA_ARCH_LIST` | CUDA architectures to compile for | `12.0` = RTX 5090, `12.1` = GB10 (DGX Spark) |

   To verify your GPU's compute capability: `python -c "import torch; print(torch.cuda.get_device_capability())"`

   If you only need to support one machine, pass a single arch to halve compile time.

## Tested configurations

| Machine | GPU | SM | CUDA | Platform | PyTorch index |
|---|---|---|---|---|---|
| DGX Spark | GB10 (Grace Blackwell) | 12.1 | 13.0 | aarch64 | `pytorch-cu130` |
| GPU rig | RTX 5090 (Blackwell) | 12.0 | 13.0 | x86_64 | `pytorch-cu130` |
