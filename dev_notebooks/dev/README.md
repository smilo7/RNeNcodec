# Developer RNeNcodec 

Blah blah blah

- Demo & audio examples: _[(link](https://animatedsound.com/rnencodec/))_
- Paper (arXiv): _(link -  coming soon!)_

---

## 0) Exporting for the Web

- a) run:
```bash
scripts/a_export_rnn_step.py
```

either editing the config and checkpoint path in the script, or providing all the args on the command line to match those used for training the checkpoint model. 

You will end up with: rnn_step.onnx

- b) then run:

```bash
scripts/b_sanity_check.py
```

again, editing the path variable inside the script if necessary.

- c) NEXT, we exprt the Encodec token->audio decoder. Run:

```bash
scripts/c_export_encodec_decoder_onnx.py
```

which should produce `encodec_decode.onnx`.

...



- d) Copy your Encodec lookup tables so modelname/artifacts (/artifacts/encodec24_codebooks.f16bin ,  /artifacts/encodec24_codebooks.meta.json)


---

## 1) Create environment

```bash
# from the repo root
micromamba create -f environment.yml --no-rc -y  
micromamba run -n rnencodec python -m ipykernel install --user \
  --name rnencodec --display-name "Python (rnencodec)"
micromamba activate rnencodec   

# (or, with conda)
# conda env create -f environment.yml
# conda activate rnencodec
# python -m ipykernel install --user --name rnencodec --display-name "Python (rnencodec)"
```

---

## 2) GPU anyone?

Inference works fine (better!) on CPU, but for training you might want to use a GPU if you have one available. Do whatever works for your card and cuda environment. For example:

```bash
# Optional GPU (Linux+NVIDIA):
micromamba install -y pytorch=2.5.* torchaudio=2.5.* torchvision=0.20.* pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge

# OR what I need for my spanking new 5090 and sm_120 requirements: 
# 1) Remove conda Torch packages (avoid mixing with pip wheels)
micromamba remove -y pytorch torchaudio torchvision pytorch-cuda || true

# 2) Install pip CUDA 12.8 wheels (these support sm_120)
python -m pip install --upgrade pip
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  
```

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
