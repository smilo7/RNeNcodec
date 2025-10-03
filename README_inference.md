# RNN EnCodec Inference — Architecture & Usage (v2)

This document describes the **refactored inference stack** and how to use it for:
- **Offline generation** (end‑to‑end audio from parameters)
- **Real‑time streaming** (hop‑based, ring‑buffered decoding; CPU‑only by policy)
- **Validation (non‑autoregressive)** and **Forced mode**
- **Notebooks & CLI**
- **Public API** (simple façade with one‑liners)

Design goals: **CPU‑only low‑latency streaming**, **no silent device moves**, strict inputs, minimal duplication, predictable APIs.

---

## TL;DR — Public API

Import just the façade functions:

```python
from inference import run_inference, run_streaming, validate_non_ar, render_forced
```

### Offline (end‑to‑end) in one call
```python
audio, sr = run_inference(
    model,            # your RNN (runs on *its* device)
    encodec_model,    # EnCodec (decodes on *its* device)
    cond_seq,         # [T,C] or None (unconditional)
    warmup_latents,   # [Tw,128] (usually in [-1,1])
    clamp_val=15.0,
    top_n=24, temperature=0.8,
    sampler="gumbel",
    include_warmup_audio=False,
)
```

- The **AR loop** runs on the **model’s device** (CUDA if your model is on CUDA).
- The **decode** happens on the **EnCodec’s device**. If they differ, only the **final latent stack** is moved for decode.

### Streaming (CPU‑only, hop + ring)
```python
def on_audio_tail(tail: np.ndarray, sr: int):  # called per hop
    portaudio_write(tail)

run_streaming(
    model, encodec_model,
    warmup_latents,                  # [Tw,128] (API scales for decode)
    ring_size_frames=25, hop_frames=1,
    on_audio_tail=on_audio_tail,
    cond_iter=None,                  # or iterator yielding per-hop [1,C]
    clamp_val=15.0, sampler="gumbel", top_n=24, temperature=0.8,
)
```
**Streaming asserts CPU**: if model or EnCodec isn’t on CPU, it raises. No hidden device moves.

### Validation (non‑AR) & Forced
```python
metrics = validate_non_ar(model, encodec_model, cond_seq, target_tokens, clamp_val=15.0)
audio, sr = render_forced(model, encodec_model, external_tokens, cond_seq=None, clamp_val=15.0)
```

---

## Directory layout

```
inference/
  core.py            # InferenceEngine (stateful), strict cond checks, out-of-place preprocess
  encodec_adapter.py # Adapter to your EnCodec helpers (codes↔latents, latents→audio), no device moves
  runners.py         # OfflineAudioRunner, StreamingTokenRunner (CPU-only), ValidationRunner, ForcedModeRunner
  api.py             # Public façade: run_inference / run_streaming / validate_non_ar / render_forced
  __init__.py        # Re-exports the façade
```

---

## Key contracts

- **Latents**: 128‑D EnCodec embeddings per frame, **native EnCodec scale** (≈ `[-clamp_val, +clamp_val]`) when stored/decoded.
- **Preprocess for RNN**: **out‑of‑place** clamp+normalize to `[-1,1]`; the stored latents remain in EnCodec scale for decode.
- **Conditioning**: concatenated `[latent(128) | cond(C)]`. **Strict**: no pad/crop. If you pass conditioning with wrong width → early `ValueError`. Unconditional (`cond_seq=None`) is **always allowed**; zeros are fed each step.
- **Sampling**: handled **inside your model** (`sample_mode ∈ {"gumbel","sample","argmax"}`, `top_n`, `temperature`). The engine does **not** re‑sample. If TF leaves a placeholder, engine uses argmax for that level to build `step_latent`.
- **Devices**:
  - **Offline/Forced/Validation**: **follow the model’s device** for AR; decode on EnCodec’s device. **No module is moved** by inference.
  - **Streaming**: **CPU‑only** (asserts if not). No device hops in the hop loop.

---

## Core components

### `InferenceEngine` (core.py)
- Tracks **hidden state** and **current latent**.
- `prime(warmup_latents, cond_seq=None)` — warms hidden using warmup latents without mutating latent scale.
- `step_autoregressive(cond_t)` — one AR step; prefers model’s `step_latent`, otherwise reconstructs from sampled tokens.
- Strict conditioning checks; unconditional allowed.

**Device rule**: `engine.device = next(model.parameters()).device`. The engine never moves your model.

### `EncodecAdapter` (encodec_adapter.py)
- Wraps your helpers: `efficient_codes_to_latents`, `latents_to_audio_simple`.
- **Never moves** the EnCodec module; records its device and moves **latents to it** right before decode.
- Preallocates a single‑frame codes buffer on EnCodec’s device for fast repeated use.

### Runners (runners.py)
- **OfflineAudioRunner** — builds `[1,128,T]` on engine.device; copies that stack to EnCodec’s device **only for decode**.
- **StreamingTokenRunner** — **asserts CPU**, preallocates a `[1,128,R]` ring + scratch window, appends `hop_frames`, reassembles via two slices, decodes full ring, and returns only the last hop’s tail.
- **ValidationRunner** — non‑AR metrics (NLL, top‑1) per codebook.
- **ForcedModeRunner** — renders audio from external token stacks `[T,n_q]` (no AR).

---

## Notebook / Script usage

**Minimal offline example**
```python
from inference import run_inference

audio, sr = run_inference(
    model, enc_model, cond_seq, warmup_latents,
    clamp_val=15.0, top_n=24, temperature=0.8, sampler="gumbel",
    include_warmup_audio=False,
)
```

**Notes**
- If `include_warmup_audio=True` and your warmup is in `[-1,1]`, the façade **scales it for decode** internally; you don’t have to.
- If `cond_size == 0`, pass `cond_seq=None`.
- If `cond_size > 0`, `cond_seq` must be `[T,C]` (or `[1,T,C]`) with exact `C` or you’ll get a clear error.

---

## Real‑time streaming (CPU‑only)

```python
from inference import run_streaming

def cond_iter():
    while True:
        yield None  # or yield a [1,C] tensor per hop

def on_audio_tail(tail: np.ndarray, sr: int):
    portaudio_write(tail)

run_streaming(
    model_cpu, encodec_cpu, warmup_latents,
    ring_size_frames=25, hop_frames=1,
    on_audio_tail=on_audio_tail, cond_iter=cond_iter(),
    clamp_val=15.0, sampler="gumbel", top_n=24, temperature=0.8,
)
```

**RT hygiene**
- Set thread caps at process start *before* `import torch`:
  - `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
  - `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`
- Keep audio callback tiny; run RNN in a separate worker thread; preallocate tensors; avoid logging/alloc in the hop loop.

---

## Troubleshooting

- **Device mismatch**: Offline follows model’s device; decode follows EnCodec’s. We never `.to(...)` your modules. If devices differ, we copy **only** the decode window.
- **Streaming error** “requires CPU devices”: Move both model and EnCodec to CPU for streaming.
- **Garbage audio**: Ensure latent history is never normalized in place; preprocessing for RNN is out‑of‑place.
- **Condition width mismatch**: strict error (no pad/crop). If unconditional, pass `cond_seq=None`.
- **“unknown mode”**: sampler must be `"gumbel"`, `"sample"`, or `"argmax"` to match your model.
- **Training error** “Inference tensors cannot be saved for backward”: don’t reuse tensors created under `torch.inference_mode()` in training. If you must, call `.clone()` before using them in a training forward.

---

## Design choices & rationale (high‑level)

- **Model‑driven sampling** keeps a single source of truth and avoids double sampling.
- **Out‑of‑place preprocess** preserves correct EnCodec‑scale latents for audio decode.
- **No hidden device moves** avoids CUDA↔CPU ping‑pong and training side effects.
- **CPU‑only streaming** provides stable latency and predictable CPU usage (thread caps), with a tensor ring + two‑slice reassembly.

---

## License & credits

Include your project license here. Credit to the original training/inference notebooks and the EnCodec authors.
