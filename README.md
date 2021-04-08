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

1. Download [LJ Speech Data](). In this example it's in `data/`

2. Make a list of the file names to use for training/testing.

   ```command
   ls data/*.wav | tail -n+10 > train_files.txt
   ls data/*.wav | head -n10 > test_files.txt
   ```
    `-n+10` and `-n10` indicates that this example reserves the first 10 audio clips for model testing.

3. Edit the configuration file and train the model.

    Below are the example commands using `flowvocoder.json`

   ```command
  
   python train.py -c configs/flowvocoder.json --tr
   ```
   Single-node multi-GPU training is automatically enabled with [DataParallel] (instead of [DistributedDataParallel] for simplicity).

   For mixed precision training, set `"fp16_run": true` on the configuration file.

   You can load the trained weights from saved checkpoints by providing the path to `checkpoint_path` variable in the config file.

   `checkpoint_path` accepts either explicit path, or the parent directory if resuming from averaged weights over multiple checkpoints.

   ### Examples
   insert `checkpoint_path: "experiments/flowvocoder/flowvocoder_5000"` in the config file then run
   ```command
   python train.py -c configs/flowvocoder.json --tr
   ```

   for loading averaged weights over 10 recent checkpoints, insert `checkpoint_path: "experiments/flowvocoder"` in the config file then run
   ```command
   python train.py -a 10 -c configs/flowvocoder.json
   ```
   
4. Synthesize waveform from the trained model.

   insert `checkpoint_path` in the config file and use `--synthesize` to `train.py`. The model generates waveform by looping over `test_files.txt`.
   ```command
   python train.py --synthesize -c configs/flowvocoder.json
   ```
   if `fp16_run: true`, the model uses FP16 (half-precision) arithmetic for faster performance (on GPUs equipped with Tensor Cores).
   
   
## Reference
NVIDIA Tacotron2: https://github.com/NVIDIA/waveglow

NVIDIA WaveGlow: https://github.com/NVIDIA/waveglow

r9y9 wavenet-vocoder: https://github.com/r9y9/wavenet_vocoder

FloWaveNet: https://github.com/ksw0306/FloWaveNet

Parakeet: https://github.com/PaddlePaddle/Parakeet

[Tacotron2]: https://github.com/NVIDIA/tacotron2
[DataParallel]: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
[DistributedDataParallel]: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
[WaveFlow]: https://arxiv.org/abs/1912.01219
[LJ Speech Data]: https://keithito.com/LJ-Speech-Dataset
[Apex]: https://github.com/nvidia/apex
[official implementation]: https://github.com/PaddlePaddle/Parakeet
