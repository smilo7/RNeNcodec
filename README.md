# RNeNcodec — QuickStart

RNeNcodec is a lightweight RNN over Encodec tokens for real-time(ish) audio generation. This repo gives you a 5-minute path to (1) run inference with a pretrained checkpoint and (2) try a tiny training loop on a small example dataset.

- Demo & audio examples: _(link)_
- Paper (arXiv): _(link)_

---

## 0) Requirements

- Python ≥ 3.9
- Conda/Mamba (recommended)
- Linux or macOS (Windows may work; real-time audio is easier on Linux/macOS)
- For real-time audio: PortAudio runtime (Linux: `sudo apt-get install -y libportaudio2`)

---

## 1) Create environment

```bash
# from the repo root
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate rnencodec              # or: conda activate rnencodec
```

If you don’t use conda, ensure Python ≥3.9 and `pip` are available in your venv.

---

## 2) Install packages (editable)

This installs **rnencodec** and the separate **rtpysynth** real-time engine (UI optional).

```bash
# repo root
pip install -e .
pip install -e ./synth[ui]
```

> Import names:
> - `rnencodec` → your package (e.g., `from rnencodec import RNNGenerator`)
> - `rtpysynth` exposes `realtime_synth` and `realtime_synth_ui` (e.g., `from realtime_synth.engine import RTStream`)

---

## 3) Get the QuickStart artifacts (weights + dataset)

This downloads the **pretrained checkpoint** and the **example HF-format dataset**, verifies SHA256, and (for the dataset) **auto-extracts** to `artifacts/data/waterfill_quickstart_hf_dataset/`.

```bash
python scripts/download_artifacts.py --all
```

You can fetch them separately:

```bash
python scripts/download_artifacts.py --weights
python scripts/download_artifacts.py --dataset
```

Paths used by the QuickStart notebooks/configs:

- Weights: `artifacts/weights/waterfill_quickstart.pt`
- Dataset (after extract): `artifacts/data/waterfill_quickstart_hf_dataset/`

---

## 4) Run the QuickStart notebooks

```bash
jupyter lab quickstart/
# open: 1_Inference.ipynb  (zero-to-audio in a few cells)
# open: 1_Train.ipynb      (tiny training loop on example dataset)
```

> Tip: the first cell in each notebook usually contains:
> ```python
> %load_ext autoreload
> %autoreload 2
> ```
> so edits to your package take effect without restarting the kernel.

---

## 5) Minimal code examples

### Inference (Python)

```python
from pathlib import Path
from rnencodec.generator.generator import RNNGenerator

ckpt = Path("artifacts/weights/waterfill_quickstart.pt")
gen = RNNGenerator.from_checkpoint(ckpt, device="cpu")   # or "cuda"
gen.prime(warmup_frames=200)

audio = gen.generate(seconds=5.0, sample_rate=24000)     # np.ndarray
```

### Real-time stream (optional)

```python
from realtime_synth.engine import RTStream

stream = RTStream(sample_rate=24000, callback=lambda n: gen.next_samples(n))
stream.start()
# ... interact ...
stream.stop()
```

### Tiny training sketch

```python
import yaml
from rnencodec.model.gru_audio_model import RNN, GRUModelConfig
# from rnencodec.audioDataLoader.audio_dataset import AudioHFLoader  # your loader

cfg = yaml.safe_load(open("rnencodec/configs/quickstart_train.yaml"))
model = RNN(GRUModelConfig(**cfg["model"]))

# ds = AudioHFLoader(root=cfg["dataset_root"], split="train")
# ... your minimal training loop here ...
```

---

## 6) Where things live

```
repo_root/
├─ rnencodec/                    # installable package
│  ├─ generator/                 # RNNGenerator, streaming helpers
│  ├─ model/                     # GRU model & config
│  ├─ audioDataLoader/           # dataloader(s)
│  ├─ utils/                     # downloads, IO, misc
│  └─ configs/                   # quickstart_{infer,train}.yaml
├─ quickstart/                   # two notebooks users should run first
│  ├─ 1_Inference.ipynb
│  └─ 1_Train.ipynb
├─ scripts/
│  └─ download_artifacts.py      # pulls weights + dataset (verifies SHA256)
├─ artifacts/                    # created on first download
│  ├─ weights/waterfill_quickstart.pt
│  └─ data/waterfill_quickstart_hf_dataset/...
└─ synth/                        # separate package (rtpysynth)
```

---

## 7) Troubleshooting

- **PortAudio missing (real-time audio):**
  - Ubuntu: `sudo apt-get install -y libportaudio2`
- **Jupyter widgets don’t display (UI extra):**
  - Ensure `ipywidgets` is installed (it is when you used `./synth[ui]`).
- **CUDA not found:** set `device: "cpu"` in `rnencodec/configs/quickstart_infer.yaml`.
- **Import errors in notebooks:** make sure you ran `pip install -e .` and selected the right kernel (the env you created).

---

## 8) Reproducibility & versions

Artifacts are served under a versioned path (e.g., `.../RNeNcodec/v0.1/...`) and verified via SHA256 in `scripts/download_artifacts.py`. To pin a new model/dataset release:

1. Upload files under a new version folder on the server (e.g., `v0.2/`).
2. Update the URLs + SHA256 constants in `scripts/download_artifacts.py`.
3. Bump any references in `rnencodec/configs/*`.

---

## 9) License & citation

- Code license: MIT _(or your choice)_
- If you use RNeNcodec in academic work, please cite: _(arXiv entry)_

```bibtex
@misc{rnencodec2025,
  title   = {RNeNcodec: Lightweight RNN over Encodec Tokens for Interactive Audio},
  author  = {Wyse, Lonce and ...},
  year    = {2025},
  eprint  = {...},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD}
}
```
