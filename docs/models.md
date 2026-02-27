# Models

All models are pure PyTorch, fully defined in `src/light_llm/models/<name>/`.

Currently implemented:

| model         | type           | location                            |
|---------------|----------------|-------------------------------------|
| `transformer` | decoder-only   | `src/light_llm/models/transformer/` |

---

## Transformer

A GPT-style autoregressive decoder with every component configurable via YAML/JSON.
The defaults follow a modern small-LLM recipe (RMSNorm + RoPE + SwiGLU + GQA).

### Quick instantiation

```python
from light_llm.models.transformer import Transformer, TransformerConfig

cfg = TransformerConfig(vocab_size=50257, d_model=512, num_layers=6)
model = Transformer(cfg)
print(model)   # prints param count and key settings
```

---

## Architecture options

### Attention — `model.attention`

| key              | default  | options                          | notes                                    |
|------------------|----------|----------------------------------|------------------------------------------|
| `variant`        | `gqa`    | `mha` `gqa` `mqa` `complex`     | see below                                |
| `num_heads`      | `8`      | int                              | query heads                              |
| `num_kv_heads`   | auto     | int                              | KV heads; auto = `num_heads // 4`       |
| `head_dim`       | auto     | int                              | auto = `d_model // num_heads`           |
| `dropout`        | `0.0`    | float                            |                                          |
| `bias`           | `false`  | bool                             | add bias to QKV + output projections     |
| `mask_type`      | `causal` | `causal` `bidirectional` `sliding_window` |                               |
| `window_size`    | —        | int                              | tokens in sliding window                 |
| `use_flash_attn` | `false`  | bool                             | requires `pip install flash-attn`        |

**Variants:**
- `mha` — standard Multi-Head Attention (`num_kv_heads == num_heads`)
- `gqa` — Grouped Query Attention; multiple query heads share each KV head (used in LLaMA 3, Mistral)
- `mqa` — Multi-Query Attention; all query heads share a single KV head
- `complex` — complex-valued attention where Q and K live in ℂ; scores are Re(Q·K̄) (Hermitian inner product); useful for studying phase-based representations

### Positional encoding — `model.positional`

| key            | default    | options                                      |
|----------------|------------|----------------------------------------------|
| `encoding`     | `rope`     | `rope` `alibi` `sinusoidal` `learned` `none` |
| `rope_base`    | `10000.0`  | float — RoPE θ base                          |
| `rope_scaling` | —          | float > 1 extends context linearly           |

- `rope` — Rotary Position Embedding. Applied inside attention to Q and K.  Most modern default.
- `alibi` — linear distance penalty added to attention logits. No learned parameters.  Good for length generalisation.
- `sinusoidal` — fixed sinusoidal added to token embeddings.
- `learned` — learnable absolute embeddings added to token embeddings.
- `none` — no positional encoding (use with caution).

When using `complex` attention, RoPE is applied as a complex phase rotation: multiplying (q_re + i·q_im) by e^{iθ}.

### Feed-forward network — `model.ffn`

| key                | default    | options                |
|--------------------|------------|------------------------|
| `variant`          | `swiglu`   | `swiglu` `geglu` `mlp` |
| `expansion_factor` | `2.6667`   | float (≈ 8/3)          |
| `dropout`          | `0.0`      | float                  |
| `activation`       | `silu`     | `silu` `gelu` `relu` (mlp only) |
| `bias`             | `false`    | bool                   |

- `swiglu` — SiLU(gate) × up, then down projection. Modern default. With expansion ≈ 8/3 it is parameter-equivalent to a 4× standard MLP.
- `geglu` — same structure but with GELU gate.
- `mlp` — two-layer MLP: Linear → activation → Linear.

### Normalisation — `model.norm`

| key         | default    | options                        |
|-------------|------------|--------------------------------|
| `type`      | `rmsnorm`  | `rmsnorm` `layernorm`          |
| `eps`       | `1e-5`     | float                          |
| `placement` | `pre`      | `pre` `post` `sandwich`        |

- `pre` — norm before sub-layer, residual outside (modern default).
- `post` — residual first, then norm (original "Post-LN" Transformer).
- `sandwich` — norm before and after the sub-layer.

### Other model-level options

| key                    | default | description                                                  |
|------------------------|---------|--------------------------------------------------------------|
| `tie_embeddings`       | `true`  | Share input embedding and LM-head weights                    |
| `parallel_layers`      | `false` | Compute attention and FFN in parallel on the same pre-norm input (PaLM-style) |
| `dropout`              | `0.0`   | Embedding dropout and residual dropout                       |
| `init_std`             | `0.02`  | Weight initialisation std                                    |
| `residual_scaled_init` | `true`  | Scale residual projection init by 1/√(2·layers) (GPT-2)    |

---

## Adding a new model

1. Create a directory `src/light_llm/models/<name>/`.
2. Add at minimum `config.py` (Pydantic config), `model.py` (nn.Module), and `__init__.py`.
3. The model only needs to expose a `forward(tokens) → logits` signature to be compatible with the Trainer.

```
src/light_llm/models/
  transformer/   ← existing
  mamba/         ← your new model
    __init__.py
    config.py
    model.py
```
