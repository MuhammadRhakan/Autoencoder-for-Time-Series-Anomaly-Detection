# Multi-Scale Temporal Convolutional Network Autoencoder

<p align="center">
    <img src="/assets/time_series_reconstruction_mechanism.gif" />
</p>

Unsupervised, reconstruction-based anomaly detection for multivariate time series in Cyber-Physical Systems, using a multi-scale Temporal Convolutional Network (TCN) autoencoder. Evaluated on the SWaT water-treatment dataset.

## Overview
The model is trained only on **normal operation**, and at inference time, abnormal behaviour shows up as a high reconstruction error. Any points are flagged as anomalies when the scores exceeds a threshold set at the *n*-th percentile of training set reconstruction errors.

### 🔑 Key Highlights
* **Multi-Scale Feature Extraction:** encoder captures short-term spikes alongside long-range temporal dependencies across sensors.
* **Temporal Convolutional Network:** causal dilated convolutions expand receptive fields without recurring memory bottlenecks.
* **Modular Pipeline:** preprocessing, windowing, training, inference, and evaluation are separate modules.

### 📈 Performance Results
| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Precision** | `0.99` | Few false positives during steady-state |
| **Recall** | `0.65` | Capture rate on stealthy bias attacks |
| **F1-Score** | `0.79` | Harmonic mean balance |
| **ROC-AUC** | `0.83` | Threshold-independent ranking quality |
| **PR-AUC** | `0.74` | Threshold-independent, reliable for class imbalance |

Precision and Recall depend on the chosen percentile threshold, while ROC and PR do not.

## Architecture
<p align ="center">
    <img src="/assets/architecture.png" />
</p>

## Project Structure
```
Multi-Scale-TCN-Autoencoder/
├── assets/                        # Image resources
├── model/
│   ├── mstcn.py                   # Model architecture
│   └── trainer.py                 # Wrapper file for modeling pipeline
├── notebooks/
│   └── data_analysis.ipynb        # Exploratory data analysis
├── src/
│   ├── callbacks.py               # Neural network callback configurations
│   ├── preprocessing.py           # Data preprocessing steps
│   ├── run_data_preparation.py    # Full data preparation execution
│   └── sliding_window.py          # Sliding window for neural network input
├── .gitignore
├── README.md                   
├── config.yaml                    
└── main.py                        # Code execution from training to evaluation
```

## Dataset
The dataset used for this experiment is **Secure Water Treatment (SWaT)** froim iTrust, STUD. The data cannot publicly be disclosed due to confidentiality, therefore a formal request must be submittedd and a local path be configured in `config.yaml`.

Request access:
https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/

The raw data consists of two xlsx files:
|File|Description|
|:----|:----|
|SWaT_Dataset_Normal_v1| 4 days normal operation for model training |
|SWaT_Dataset_Attack_v0| 7 days attacked operation for validation and testing |

## Execution Sample
Run data preparation
```
python -m src.run_data_preparation --config "config.yaml"
```

Execute main script.py
```
python main.py --mode train
```

## 📚 References
* J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, “A Dataset to Support Research in the Design of Secure Water Treatment Systems.”
* X. Ma, F. Wu, J. Yue, P. Feng, X. Peng, and J. Chu, “MSE-TCN: Multi-scale temporal convolutional network with channel attention for open-set gas classification,” Microchemical Journal, vol. 207, Dec. 2024.
