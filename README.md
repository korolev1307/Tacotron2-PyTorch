# Tacotron2-PyTorch
Yet another PyTorch implementation of [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). The project is highly based on [these](#References). I made some modification to improve speed and performance of both training and inference.
Also, optimized for actual version of libraries such as torch and librosa, and using Russian language with phonemes.

## Preprocessing
Currently only support [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) format dataset. You can modify `hparams.py` for different sampling rates. `prep` decides whether to preprocess all utterances before training or online preprocess. `pth` sepecifies the path to store preprocessed data. 
But you can use own dataset in the same format as [LJ Speech](https://keithito.com/LJ-Speech-Dataset/).

##Phonemizer 
Currently used and test phonemizer [XPhoneBERT](https://github.com/thelinhbkhn2014/Text2PhonemeSequence). You should preprocess your dataset and replace graphemes by their phonemes. 

## Training
1. For training Tacotron2, run the following command.
```bash
python3 train.py \
    --data_dir=<dir/to/dataset> \
    --ckpt_dir=<dir/to/models>
```

2. If you have multiple GPUs, try [distributed.launch](https://pytorch.org/docs/stable/distributed.html#launch-utility).
```bash
torchrun --nnodes <NUM_NODES> --nproc_per_node <NUM_GPUS> train.py \
    --data_dir=<dir/to/dataset> \
    --ckpt_dir=<dir/to/models>
```
Note that the training batch size will become <NUM_GPUS> times larger.

3. For training using a pretrained model, run the following command.
```bash
python3 train.py \
    --data_dir=<dir/to/dataset> \
    --ckpt_dir=<dir/to/models> \
    --ckpt_pth=<pth/to/pretrained/model>
```

4. For using Tensorboard (optional), run the following command.
```bash
python3 train.py \
    --data_dir=<dir/to/dataset> \
    --ckpt_dir=<dir/to/models> \
    --log_dir=<dir/to/logs>
```
You can find alinment images and synthesized audio clips during training. The text to synthesize can be set in `hparams.py`.

## Ground Truth Aligned
You can make GTA, by using `GTA.py` file. This process described in [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf).


- For making GTA use this command:
```bash
python3 GTA.py \
    --ckpt_pth=<pth/to/load/data> \
    --mel_pth=<pth/to/save/mels> \
    --data_dir=<pth/to/load/data> \
```
## Inference
- For synthesizing wav files, run the following command.

```bash
python3 inference.py \
    --ckpt_pth=<pth/to/model> \
    --img_pth=<pth/to/save/alignment> \
    --npy_pth=<pth/to/save/mel> \
    --wav_pth=<pth/to/save/wav> \
    --text=<text/to/synthesize>
```

## Vocoder
A vocoder is implemented. Using [Hifi-GAN](https://github.com/jik876/hifi-gan). Check the README from hifigan folder.

## References
This project is highly based on the works below. Mainly on first work.
- [Tacotron2 by BogiHsu](https://github.com/BogiHsu/Tacotron2-PyTorch)
- [Tacotron2 by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron by keithito](https://github.com/keithito/tacotron)
