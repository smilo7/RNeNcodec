# RNNControl

Jupyter notebooks that run experiments investigating the effect of conditional parameters on audio synthesis.  

Input to the RNN is a sequence of vectors where one component represents the audio signal and the others represent conditional parameters. The audio signal is mulaw encoded (to [0,255]) and then mapped to [0,1]. 

During Training, the output is a vector of logits, one for each possible mu-coded value ( in [0,255]). That is, it is a considered a category and compared to the one-hot representation of the target. 

During inference, the logits are converted to a probability distribution with softmax, a sample is chosen, and then it's index (a mu-law value) is converted to a single floating point value as the audio input for the next sequence step. 

The default model is very small (4 layers of 48 units), with about 70K trainable parameters, and trains (recognizably) on the nsynth64.76_sm dataset (provided) of 260 2.5 sec. sounds in about 12 minutes (40 epochs of 100 batches of 256 sequences) on an nvidia 5090 (about 3 times longer on the desktop CPUs). Slow on inference, generating 2 secs of sound in 9 seconds on the GPU, or 6 secs on the CPUs (yep!).

## Quick Start 

### Prerequisites
- **Conda** (Miniconda or Anaconda)
- (Optional) **NVIDIA GPU** with a recent driver if you want CUDA acceleration

```bash
# 1) Create the environment (CPU by default)
conda env create -f environment.yml

# 2) Activate it
conda activate basicaudio

# 3) (Optional) Register a Jupyter kernel with this env
python -m ipykernel install --user --name basicaudio
```
### (Optional) NVIDIA GPU Acceleration

By default, the env is **CPU-only** (portable).  
On machines with a compatible NVIDIA driver, install CUDA support with one command:

```bash
# With the env active
conda install -n basicaudio -c nvidia pytorch-cuda=xx.x
```

Install the small test dataset [nsynth.64.76_sm](https://drive.google.com/file/d/1zzXccguczXJIDh8vxoXvt1xGmKlGn_f6/view?usp=sharing) into the /data folder.  (Check to make sure the wav files are are ./data/nsynth.64.76_sm and not in a nested directory).

Open  jupyter lab,   

1. Train the model:
   - Open `Train.ipynb` and run the cells to train the model.

2. Generate audio samples:  
   - Open `Inference.ipynb` 
   - update the configuration structure to point to the path where the trained model was written,
   - run the cells to generate audio samples using the trained model.


## Folder structure
**Modules**  
+--model  
   |- gru_audio_model
+--utils 
    |- utils.py
+--audioDataLoader 
    |- audio_dataset.py
    |- mulaw.py  
inference.py


**Files**  
* Train.ipynb: for training and saving model (an RNN)
* Inference.ipynb: for loading model and generating audio samples

**Data**  
* data: directory containing audio files for training
This model was developed using a modified subset of the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth). That subset can be found here: [nsynth.64.76_sm](https://drive.google.com/file/d/1zzXccguczXJIDh8vxoXvt1xGmKlGn_f6/view?usp=sharing). There are two instruments (a clarinet and a trumpet) sampled over an octave of notes (midi pitch 64-76) and 10 synthetically imposed amplituide values. Each file is the middle 2.5 seconds of the original sample (no attack and decay segments). The parameter values are in the file names and extracted by the data loader in order to provide them to the model for training. You must specify the parameter names and the range of values you want to map them to in the configuration files provided to the Dataset and the Model (see Train.ipynp and Inference.ipynb). 

### Preparing your own data
- Wav files should all be in a single folder
- each should correspond to a single conditional parameter configuration
- the parameter used for training (and later controling) the network should be contained in the file names (because that's where the Data Loader looks for them) in the form _PNAMExx.xx where xx.xx is a floating point parameter value for the parameter PNAME.
- - e.g. For a file named nsynth_instID02.00_p76.00_a00.30.wav , the data loader will find three parameters, instID, p, and a. 
- - These parameters are listed in the DataSet configuration files along with the mappings from the values found in the file names (arbitrary) to the values used to train the net (typically [0,1]). In our example dataset, the midi Pitch values are in the file names, and [64, 76] maps to [0,1] for training (and inference). 


**Authors**  
* Lonce Wyse <lonce.wyse@upf.edu>



