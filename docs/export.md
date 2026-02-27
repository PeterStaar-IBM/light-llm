# llm-export

Export a trained checkpoint to a portable inference format.

Supported formats:

| format         | use case                                                 |
|----------------|----------------------------------------------------------|
| `safetensors`  | Share weights, convert between frameworks               |
| `onnx`         | Deploy with ONNX Runtime, TensorRT, or CoreML           |
| `torchscript`  | Deploy without a Python install via `torch::jit::load`  |

---

## SafeTensors

Re-export weights only (strips optimizer state, useful for sharing):

```bash
llm-export \
    --checkpoint checkpoints/small/best \
    --format     safetensors \
    --output     exports/small-weights.safetensors
```

---

## ONNX

```bash
llm-export \
    --checkpoint checkpoints/small/best \
    --format     onnx \
    --output     exports/small.onnx \
    --seq-len    128        # tracing sequence length
    --dtype      float32    # ONNX works best in float32
```

Batch and sequence axes are exported as dynamic, so the exported graph accepts any input shape at runtime.

After export the tool automatically verifies that the ONNX Runtime output matches the PyTorch output (requires `onnxruntime`):

```
✓ Verification passed  (max diff: 0.000012)
```

Run inference with ONNX Runtime:

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("exports/small.onnx")
input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)   # [batch, seq]
logits = sess.run(["logits"], {"input_ids": input_ids})[0]
```

---

## TorchScript

```bash
llm-export \
    --checkpoint checkpoints/small/best \
    --format     torchscript \
    --output     exports/small.pt
```

Load and run in C++ or Python without the full `light_llm` package:

```python
import torch

model = torch.jit.load("exports/small.pt")
model.eval()

input_ids = torch.randint(0, 50257, (1, 128))
logits = model(input_ids)   # [1, 128, vocab_size]
```

---

## All options

| option         | default      | description                                              |
|----------------|--------------|----------------------------------------------------------|
| `--checkpoint` | required     | Checkpoint path or tag                                   |
| `--format`     | `onnx`       | `onnx` / `torchscript` / `safetensors`                  |
| `--output`     | `exports/model` | Output file path (extension added automatically)      |
| `--seq-len`    | `128`        | Sequence length for tracing (onnx / torchscript)         |
| `--batch-size` | `1`          | Batch size for tracing                                   |
| `--dtype`      | `float32`    | Export dtype (`float32` recommended for ONNX)            |
| `--opset`      | `17`         | ONNX opset version                                       |
| `--verify`     | `true`       | Verify ONNX output against PyTorch after export          |
