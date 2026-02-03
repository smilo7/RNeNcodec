# RNeNcodec — QuickStart

RNeNcodec is a lightweight RNN over Encodec tokens for real-time(ish) audio generation. This repo gives you a 5-minute path to (1) run inference with a pretrained checkpoint and (2) try a tiny training loop on a small example dataset.

- Demo & audio examples: _[(link](https://animatedsound.com/rnencodec/))_
- Paper (arXiv): _(link -  coming soon!)_

---

## 0) Requirements

- Python ≥ 3.9
- Conda (recommended)
- Linux or macOS (Windows may work; real-time audio is easier on Linux/macOS)
- For real-time audio: PortAudio runtime (Linux: `sudo apt-get install -y libportaudio2`)
- Git (sudo apt-get install -y git) because you'll be installing a synth from a git repo

---

## 1) Create environment

```bash

conda env create -f environment.yml 
conda activate rnencodec 
python -m pip install -r requirements-conda.txt 
python -m pip install -e .
python -m ipykernel install --user --name rnencodec --display-name "Python (rnencodec)"
```

#### plan B (to create environment only if you don't use Conda)
Conda/Mamba is strongly recommended because it installs both Python deps *and* native libraries reproducibly. If you can’t or don’t want to use conda/mamba, you can use pip + a system package manager instead.
#### plan B example: macOS (Homebrew + venv)

```bash
# system deps
brew install ffmpeg libsndfile portaudio git-lfs

# create a venv (from repo root)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# python deps
pip install -r requirements.txt
pip install "rtpysynth[ui] @ git+https://github.com/lonce/RTPySynth@v0.1.4"
pip install -e .

# (optional) register kernel for Jupyter
python -m ipykernel install --user --name rnencodec --display-name "Python (rnencodec)"
```
---

If you happen to like to run your jupyter notebooks from inside VSCode, then there is a ["renderer" extension from microsoft](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter-renderers) that you might want so that you can render the interactive synth that appears in the inferencing notebooks): 

---
## 2) GPU anyone?

Inference works fine (better!) on CPU, but for training you might want to use a GPU if you have one available. Do whatever works for your card and cuda environment. For example:

```bash
# Optional GPU (Linux+NVIDIA):
micromamba install -y pytorch=2.5.* torchaudio=2.5.* torchvision=0.20.* pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge

# OR what I need for my spanking new 5090 and sm_120 requirements: 
# 1) Remove conda Torch packages (avoid mixing with pip wheels)
micromamba remove -y pytorch torchaudio torchvision pytorch-cuda || true
# or
pip uninstall -y torch torchvision torchaudio

# 2) Install pip CUDA 12.8 wheels (these support sm_120)
python -m pip install --upgrade pip
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  
```

---

## 3) Get the QuickStart waterfill dataset 

Users are expected to have datasets of audio files, each with a corresponding .csv file that has parameters in columns (with parameter names as headers) at a sample rate of 75 frames per second (see 1_dataset.ipynb notebook for details). We assume this format is easy for users to create, and it is easily readable. It accommodates parameters that are dynamic  throughout a dataset sample, but also (with redundancy) static parameters. The 1_dataset.ipynb notebook then reads a dataset in this format to prepare it in a more efficient form that is ready for data loading by RNeNcodec. One such dataset already in the "user format" is **water_fill**. 

This downloads the **water_fill user-formatted datset** from Hugging Face into the data/ directory. 

```bash
cd data/ # data/ is just the default location
python -m pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lonce/quickstartdata', repo_type='dataset', local_dir='quickstartdata')"
```

- You are now ready to run the 1_dataset.ipynb to prepare the dataset for loading by RNeNcodec. 

---

## 4) Run the QuickStart notebooks

```bash
jupyter lab quickstart/
# open: Inference.ipynb  (realtime synth with a widget interface)
# open: Train.ipynb      (training loop on the example dataset)
```

> Tip: the first cell in each notebook usually contains:
> ```python
> %load_ext autoreload
> %autoreload 2
> ```
> so edits to your package take effect without restarting the kernel.

---



## 5) Where things live

```
repo_root/
├─ rnencodec/                    # installable package
│  ├─ generator/                 # RNNGenerator, streaming helpers
│  ├─ model/                     # GRU model & config
│  ├─ audioDataLoader/           # dataloader(s)
│  ├─ utils/                     # downloads, IO, misc
├─ quickstart/                   # two notebooks users can run first
│  ├─ Inference.ipynb
│  └─ Train.ipynb
├─ scripts/
│  └─ download_artifacts.py      # pulls quickstarts weights + dataset (verifies SHA256)
├─ artifacts/                    # created on first download
│  ├─ weights/waterfill_quickstart.pt
│  └─ data/waterfill_quickstart_hf_dataset/...
└─ notebooks/                    # training, inference, visualization ... you'll need your own data
```

---

## 6) Troubleshooting

- **PortAudio missing (real-time audio):**
  - Ubuntu: `sudo apt-get install -y libportaudio2`
- **Jupyter widgets don’t display (UI extra):**
  - Ensure `ipywidgets` is installed (it is when you use `./synth[ui]`).

---



## 7) License & citation

- Code license: MIT 
- If you use RNeNcodec in academic work, please cite: _(arXiv entry)_

```bibtex
@misc{rnencodec2025,
  title   = {RNeNcodec: Lightweight RNN over Encodec Tokens for Interactive Audio},
  author  = {Wyse, Lonce},
  year    = {2025},
  eprint  = {...},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD}
}
```
