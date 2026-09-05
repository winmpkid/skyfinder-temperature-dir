# SkyFinder Temperature Prediction with Deep Imbalanced Regression

This project adapts **Label Distribution Smoothing (LDS)** from *Delving into
Deep Imbalanced Regression* to predict ambient temperature from SkyFinder
images. It compares the same ResNet-18 regressor trained with ordinary L1 loss
and LDS-weighted L1 loss.

The main result is deliberately mixed: LDS slightly improves validation error,
but performs worse on the colder chronological test split. This highlights the
difference between correcting label imbalance and handling temporal
distribution shift.

> This is an independent implementation of a public take-home exercise. It is
> not an official UCLA or Health Intelligence Lab project.

## Results

All errors are in degrees Celsius. The reported checkpoints were selected by
unweighted validation MAE.

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Training-median baseline | 10.226 | 11.010 | 5.945 | 7.803 |
| ResNet-18 baseline | 2.677 | 3.459 | **4.922** | **6.293** |
| ResNet-18 with LDS | **2.610** | **3.312** | 6.315 | 7.551 |

LDS reduced validation MAE by 0.066 °C (2.5%), with its largest validation
gains in the rare sub-zero ranges. Test MAE increased by 1.393 °C. The test
images were substantially colder on average than the training images, and the
LDS model showed a larger positive prediction bias. The experiment therefore
does not support claiming a general improvement from LDS on this subset.

![Baseline test temperature analysis](results/baseline_test_temperature_analysis.png)

The complete aggregate metrics, temperature-bin tables, training histories,
and plots are in [`results/`](results/). The concise experiment report is in
[`reports/SkyFinder_DIR_Brief_Report.pdf`](reports/SkyFinder_DIR_Brief_Report.pdf).

## Method

- **Data:** 2,547 valid images from SkyFinder cameras 858, 3888, and 4795.
- **Target:** `TempM` ambient temperature in degrees Celsius.
- **Split:** per-camera chronological split by capture date: 1,777 train, 375
  validation, and 395 test images. A date never appears in multiple splits.
- **Model:** ImageNet-pretrained ResNet-18 with a one-output regression head.
- **Training:** AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 32,
  20 epochs, seed 42.
- **LDS:** 1 °C label bins, Gaussian kernel size 5, sigma 2.0, inverse smoothed
  density weights normalized to mean 1.
- **Evaluation:** ordinary MAE and RMSE; LDS weights are used only in the
  training objective.

This repository implements LDS only. Feature Distribution Smoothing (FDS),
the two-stage RRT procedure, and the official DIR training schedule are outside
the present experiment.

## Repository structure

```text
.
├── src/
│   ├── prepare_data.py  # join images to weather labels and create splits
│   ├── dataset.py       # PyTorch Dataset, transforms, and data loaders
│   ├── model.py         # ResNet-18 temperature regressor
│   ├── utils.py         # LDS kernel, weights, and weighted L1 loss
│   ├── train.py         # baseline and LDS training
│   └── evaluate.py      # metrics, constant baseline, bins, and plots
├── tests/               # unit tests for the LDS utilities
├── scripts/             # source used to build the concise report
├── results/             # tracked aggregate outputs from the reported runs
├── reports/             # concise PDF report
├── datasets/            # local SkyFinder files; not committed
├── data/                # generated manifest; not committed
└── checkpoints/         # generated model weights; not committed
```

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/winmpkid/skyfinder-temperature-dir.git
cd skyfinder-temperature-dir
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The training script automatically chooses CUDA, Apple Silicon MPS, or CPU in
that order.

## Prepare the data

The dataset is not included. Download the individual SkyFinder camera images
and metadata from the [official SkyFinder dataset
page](https://mvrl.cse.wustl.edu/datasets/skyfinder/), then arrange them as:

```text
datasets/
├── complete_table_with_mcr.csv
└── images/
    ├── 858/
    │   └── *.jpg
    ├── 3888/
    │   └── *.jpg
    └── 4795/
        └── *.jpg
```

Generate the cleaned, chronological manifest:

```bash
python src/prepare_data.py
```

The manifest stores project-relative image paths, so it remains valid when the
repository is moved as a unit. Two truncated JPEGs encountered in this subset
are excluded explicitly during preparation.

## Train

Baseline:

```bash
python src/train.py
```

LDS-weighted model:

```bash
python src/train.py --lds
```

Useful experiment controls include `--epochs`, `--batch-size`,
`--learning-rate`, `--lds-bin-width`, `--lds-kernel-size`, `--lds-sigma`, and
`--run-name`. Run `python src/train.py --help` for the complete interface.

Checkpoints are intentionally not committed: each is about 128 MB and can be
recreated with the commands above.

## Evaluate

Evaluate both checkpoints on validation and test splits:

```bash
python src/evaluate.py --checkpoint best_resnet18.pt \
  --output-prefix baseline --model-label "ResNet-18 baseline" --split val

python src/evaluate.py --checkpoint best_resnet18.pt \
  --output-prefix baseline --model-label "ResNet-18 baseline" --split test

python src/evaluate.py --checkpoint best_resnet18_lds.pt \
  --output-prefix lds --model-label "ResNet-18 with LDS" --split val

python src/evaluate.py --checkpoint best_resnet18_lds.pt \
  --output-prefix lds --model-label "ResNet-18 with LDS" --split test
```

Each run writes overall metrics, per-image predictions, 5 °C temperature-bin
statistics, a training-median comparison, and an analysis plot. Per-image
prediction files remain untracked because they include local paths.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Rebuild the report

The report source is generated as an editable Word document:

```bash
python -m pip install -r requirements-report.txt
python scripts/build_brief_report.py
```

Export the generated DOCX to PDF with Microsoft Word or LibreOffice if the
public PDF needs to be refreshed.

## Limitations and next steps

- Tune LDS bin width, kernel size, sigma, and a maximum weight cap. The current
  maximum training weight is about 24.1.
- Add FDS and compare baseline, LDS, FDS, and LDS + FDS under the same protocol.
- Add more cameras and seasons while preserving leakage-safe temporal splits.
- Repeat experiments across seeds and report uncertainty.
- Analyze time of day, weather conditions, and camera-specific bias.

## Attribution

The LDS design is based on:

> Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, and Dina Katabi. “Delving
> into Deep Imbalanced Regression.” ICML, 2021.

- [Paper](https://proceedings.mlr.press/v139/yang21m.html)
- [Official implementation](https://github.com/YyzHarry/imbalanced-regression)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

SkyFinder should be cited as:

> Radu Bogdan Mihail, Scott Workman, Zach Bessinger, and Nathan Jacobs. “Sky
> Segmentation in the Wild: An Empirical Study.” WACV, 2016.

The dataset itself is not redistributed here. This repository's original code
is released under the [MIT License](LICENSE).
