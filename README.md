## FlowVocoder: A small Footprint Neural Vocoder based Normalizing Flow forSpeech Synthesis


## Setup

1. Clone this repo and install requirements

   ```command
   git clone https://github.com/tienmanhptit1312/FlowVocoder.git
   cd FlowVocoder
   pip install -r requirements.txt
   ```

2. Install [Apex](https://github.com/NVIDIA/apex) for mixed-precision training: 


## Train your model

1. Download [LJ Speech Data](https://keithito.com/LJ-Speech-Dataset/). Then, uncompress LJ-Speech dataset where you downloaded it.
2. Copy wave files from LJ-Speech directory to FlowVocoder directory.

   ```
   cp -r [LJ-Speech dataset's directory]/wavs [FlowVocoder's directory]
   ```

3. Make a list of the file names to use for training/testing.

   ```command
   ls wavs/*.wav | tail -n+1310 > train_files.txt
   ls wavs/*.wav | head -n1310 > test_files.txt
   ```
   `-n1310` indicates that this example reserves the first 1310 audio clips for model testing. The remaining dataset is used for training.

4. Edit the configuration file and train the model.

    Below are the example commands using `flowvocoder.json`

   ```command
   python train.py -c configs/flowvocoder.json --tr
   ```
   Single-node multi-GPU training is automatically enabled with [DataParallel] (instead of [DistributedDataParallel] for simplicity).

   For mixed precision training, set `"fp16_run": true` on the configuration file.

   You can load the trained weights from saved checkpoints by providing the path to `checkpoint_path` variable in the config file.

   `checkpoint_path` accepts either explicit path, or the parent directory if resuming from averaged weights over multiple checkpoints.
   It takes about a week to train this model with two V100 Nvidia GPUs with batch-size=2. You can download our pretrained model for about 1M training iterations: [link](https://drive.google.com/file/d/1K-NAXjh9DvBEiAXQHay5jC-oivMgX7RQ/view?usp=sharing) for reproducing purpose.
   
   ### Examples
   insert `checkpoint_path: "experiments/flowvocoder/flowvocoder_5000"` in the config file then run
   ```command
   python train.py -c configs/flowvocoder.json --tr
   ```

   for loading averaged weights over 10 recent checkpoints, insert `checkpoint_path: "experiments/flowvocoder"` in the config file then run
   ```command
   python train.py -a 10 -c configs/flowvocoder.json
   ```
   
5. Synthesize waveform from the trained model.

   insert `checkpoint_path` in the config file and use `--synthesize` to `train.py`. The model generates waveform by looping over `test_files.txt`.  
   ```command
   python train.py --synthesize -c configs/flowvocoder.json
   ```
   if `fp16_run: true`, the model uses FP16 (half-precision) arithmetic for faster performance (on GPUs equipped with Tensor Cores).
   
   
## Reference
NVIDIA Tacotron2: https://github.com/NVIDIA/tacotron2

WaveFlow: https://github.com/L0SG/WaveFlow

