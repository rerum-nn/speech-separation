# Audio-Visual Speech Separation with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

PyTorch implementation of DPRNN and RTFS-Net models for Audio-Visual Speech Separation task. Also repository consist our custom realisation of RTFS-Net -- RTFS-U-Net.

See the task assignment [here](https://github.com/markovka17/dla/tree/2025/project_avss).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rerum-nn/speech-separation.git
   cd asr-rnn-t
   ```

2. Create conda environment (strongly recommended):
   ```bash
   conda create -n dla python=3.10
   conda activate dla
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

### Train model

To train a model, run the following command:

```bash
$ python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
$ python3 inference.py datasets=inference dataloader=main inferencer.save_path=/save/path inferencer.from_pretrained=/model/path model=rtfs_u_net_tiny
```

The results will be saved in next path:
` ROOT_PATH/data/saved/{inferencer.save_path}`

We use next configs with implemented models:
- dprnn.yaml
- rtfs_net.yaml

All the models configs can be found in `src/configs/model`.

### Calc metric

To calc metric

```bash
$ python3 calc_metrics.py paths.mix=/path/to/mix paths.ground_truth=/path/to/ground_truth paths.predictions=/path/to/predictions
```

## How to load checkpoint

To load the final checkpoint, run the following command:

```bash
$ ./download_yadisk.sh https://disk.360.yandex.ru/d/LuYNtQZKVJPvGw content/avss/model.pth
```

Load checkpoint without validation in train set:
 ```bash
$ ./download_yadisk.sh https://disk.360.yandex.ru/d/7Z92QjhOtcuYRw content/avss/model.pth
```

Also download video_encoder checkpoint:
```bash
$ gdown 179NgMsHo9TeZCLLtNWFVgRehDvzteMZE -O data/video_encoder/lrw_resnet18_dctcn_video.pth
```

## How to load your dataset

Your dataset should be placed on yandex disk and has the next structure:
```
NameOfTheDirectoryWithUtterances
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   ├── s1 # ground truth for the speaker s1, may not be given
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   └── s2 # ground truth for the speaker s2, may not be given
│       ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│       ├── FirstSpeakerID2_SecondSpeakerID2.wav
│       .
│       .
│       .
│       └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── mouths # contains video information for all speakers
    ├── FirstOrSecondSpeakerID1.npz # npz mouth-crop
    ├── FirstOrSecondSpeakerID2.npz
    .
    .
    .
    └── FirstOrSecondSpeakerIDn.npz. 
```


```bash
$ ./download_dataset.sh dataset_link_on_yadisk content/datasets/custom_dataset
```

## How to reproduce the best result

```bash
$ python3 train.py -cn=rtfs_net trainer.override=True writer.run_name=rtfs-u-net-tiny-two-targets-long dataloader.batch_size=16 trainer.epoch_len=1250 trainer.n_epochs=150 datasets=main metrics=train model=rtfs_net_tiny model.compression_blocks=2 +trainer.use_amp_bf16=True +trainer.use_custom_init=True lr_scheduler.pct_start=0.01 trainer.log_step=250
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
