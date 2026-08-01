# FFT-LCNN for ASVspoof 2019 LA

## About

This repository implements a neural-network countermeasure for logical-access (LA) speech spoofing detection on the ASVspoof 2019 dataset.

The model uses log-power FFT spectrograms as input features and a Light Convolutional Neural Network (LCNN) with Max-Feature-Map (MFM) activations for binary spoofing detection.

Training is performed using A-Softmax loss and evaluated using Equal Error Rate (EER) metric.

## Installation

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## Dataset

Download or attach the Kaggle dataset `awsaf49/asvpoof-2019-dataset`.

By default, the configuration expects the dataset at:

```
/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/
```

To use another location, specify `ASVSPOOF_ROOT`:

```bash
ASVSPOOF_ROOT=/path/to/asvpoof-2019-dataset python3 train.py
```

## Training

Start training with the default configuration:

```bash
python3 train.py -cn=baseline
```

Hydra arguments can be used to override configuration values:

```bash
python3 train.py trainer.n_epochs=15 dataloader.batch_size=32
```

Checkpoints are saved in the `saved/` directory. The best checkpoint is stored as `model_best.pth`.

## Inference

Run evaluation using the checkpoint specified in `src/configs/inference.yaml`:

```bash
python3 inference.py
```

To specify another checkpoint:

```bash
python3 inference.py inferencer.from_pretrained=/path/to/model_best.pth
```

Predictions are saved to `data/saved/asvspoof/eval/`.

## Credits

This repository is based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
