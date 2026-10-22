# Keyword-Spotting Model Training (MobileNetV2 + Transfer Learning)

Training pipeline for the model deployed in **[ges_iavoz](https://github.com/XavierPonsM/ges_iavoz)** — real-time keyword spotting on an ESP32-S3 microcontroller. Part of my bachelor's thesis *Artificial Intelligence to the End-Device* (UPC · ETSETB, 2023).

## Approach

| Step | Detail |
|---|---|
| Dataset | Google Speech Commands — keywords *marvin, happy, eight, house* + *unknown* |
| Features | Mel-filterbank energies (MFE), 99 frames × 40 bins |
| Model | MobileNetV2 (α = 0.35) pretrained on keyword spotting, frozen backbone + new classification head (GlobalAveragePooling → Dropout → Dense/softmax) |
| Training | Nadam optimizer; final run: learning rate 0.0025, dropout 0.15, 150 epochs on Google Colab GPU |
| Export | int8 full-integer quantization to TFLite, then converted to a C array for the firmware |

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **97.18 %** |
| Test accuracy (confusion matrix) | 94.48 % |
| On-device false positives (20 h of speech) | 0.05 / h |
| On-device inference time | 299 ms |

## Repository layout

```
├── train.py                        # Training script (adapted from Edge Impulse's Keras pipeline)
├── train_input.json                # Classes, input shape and quantization settings
├── TFG_MobileNetEdgeImpulse.ipynb  # Colab notebook: dataset prep, training runs, plots, confusion matrix
├── resources/libraries/            # Edge Impulse helper libraries used by train.py
├── Dockerfile                      # Container used by Edge Impulse custom-learning blocks
└── requirements.txt
```

## Usage

The script expects the features already extracted as NumPy arrays (`X_split_train.npy`, `Y_split_train.npy`, …). They are not included in the repo because of their size (~170 MB).

```bash
pip install -r requirements.txt
python train.py \
  --data-directory ./data \
  --out-directory ./output \
  --epochs 150 \
  --learning-rate 0.0025
```

The pretrained MobileNetV2 weights are downloaded automatically from Edge Impulse's CDN on first run. Paths in `train.py` and the notebook point to the Google Drive layout used during the thesis and need to be adjusted for local use.

## Credits

Training script and helper libraries are based on [Edge Impulse](https://edgeimpulse.com)'s open Keras training pipeline. Thesis supervised by Prof. Josep Paradells (UPC).
