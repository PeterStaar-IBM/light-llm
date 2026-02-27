# Tokenizers

All tokenizers share the same `BaseTokenizer` interface so the data pipeline and tools are tokenizer-agnostic.

```python
from light_llm.tokenizer import HFTokenizer, BPETokenizer, CharTokenizer

tok = HFTokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello world")       # list[int]
text = tok.decode(ids)                 # str
print(tok.vocab_size)                  # 50257
```

---

## HFTokenizer — reuse any existing vocabulary

Wraps any HuggingFace `PreTrainedTokenizer`.  Use this when you want the same tokenization as an existing open-source model.

```python
from light_llm.tokenizer import HFTokenizer

# By HF model name (downloads on first use, cached locally)
tok = HFTokenizer.from_pretrained("gpt2")
tok = HFTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tok = HFTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# From a local directory previously saved with .save()
tok = HFTokenizer.from_pretrained("tokenizer/my-saved-tok")

# Save for offline / reproducible use
tok.save("tokenizer/gpt2-local")
tok2 = HFTokenizer.load("tokenizer/gpt2-local")
```

Access the underlying HuggingFace tokenizer for advanced use:

```python
hf_tok = tok.backend   # PreTrainedTokenizerFast instance
```

---

## BPETokenizer — train a custom vocabulary

Byte-level BPE (same backend as GPT-2) trained on your own corpus.

```python
from light_llm.tokenizer import BPETokenizer

# Train from plain-text files
tok = BPETokenizer.train(
    files=["data/raw/train.txt", "data/raw/extra.txt"],
    vocab_size=16_000,
    save_path="tokenizer/bpe-16k",   # optional: auto-save
)

# Load later
tok = BPETokenizer.load("tokenizer/bpe-16k")
```

Or via the CLI:

```bash
llm-preprocess --input data/raw/ --output data/tokens \
               --tokenizer bpe --vocab-size 16000 \
               --save-tokenizer tokenizer/bpe-16k
```

---

## CharTokenizer — character-level

Vocabulary is the set of unique characters in the training corpus.  Good for tiny experiments.

```python
from light_llm.tokenizer import CharTokenizer

tok = CharTokenizer.from_text(open("data/input.txt").read(), save_path="tokenizer/char")
tok = CharTokenizer.load("tokenizer/char")
```

---

## Interface reference

Every tokenizer implements:

| method / property      | description                                              |
|------------------------|----------------------------------------------------------|
| `vocab_size`           | Number of tokens in the vocabulary                       |
| `bos_token_id`         | Beginning-of-sequence id (or `None`)                    |
| `eos_token_id`         | End-of-sequence id (or `None`)                          |
| `pad_token_id`         | Padding id (or `None`)                                  |
| `unk_token_id`         | Unknown token id (or `None`)                            |
| `encode(text, add_special_tokens=True)` | `str → list[int]`              |
| `decode(ids, skip_special_tokens=True)` | `list[int] → str`              |
| `encode_batch(texts)` | Encode a list of strings                                 |
| `decode_batch(batch)` | Decode a list of id lists                                |
| `iter_chunks(text, chunk_size)` | Encode and yield fixed-size windows           |
| `save(path)`          | Persist to a directory                                   |
| `load(path)` (classmethod) | Load from a directory                             |

---

## Using a custom tokenizer

Subclass `BaseTokenizer` and implement the abstract methods:

```python
from light_llm.tokenizer.base import BaseTokenizer
from pathlib import Path

class MyTokenizer(BaseTokenizer):
    def __init__(self, vocab: dict[str, int]) -> None:
        self._v = vocab
        self._r = {i: t for t, i in vocab.items()}

    @property
    def vocab_size(self) -> int: return len(self._v)
    @property
    def bos_token_id(self): return self._v.get("<bos>")
    @property
    def eos_token_id(self): return self._v.get("<eos>")
    @property
    def pad_token_id(self): return self._v.get("<pad>")
    @property
    def unk_token_id(self): return self._v.get("<unk>")

    def encode(self, text, add_special_tokens=True):
        return [self._v.get(c, self.unk_token_id) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._r.get(i, "") for i in ids)

    def save(self, path):
        import json
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "vocab.json").write_text(json.dumps(self._v))

    @classmethod
    def load(cls, path):
        import json
        vocab = json.loads((Path(path) / "vocab.json").read_text())
        return cls(vocab)
```
