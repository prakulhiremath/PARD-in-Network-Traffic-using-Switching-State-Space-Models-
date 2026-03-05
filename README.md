# Probabilistic Attack Regime Detection in Network Traffic
## Using Switching State-Space Models (SSSM)

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Domain](https://img.shields.io/badge/domain-cybersecurity-red.svg)
![ML](https://img.shields.io/badge/research-machine%20learning-orange.svg)
![Status](https://img.shields.io/badge/status-active%20development-yellow.svg)

A probabilistic framework for **early detection of cyber-attack stages** in network telemetry
using **Switching State-Space Models (SSSM)**. The system models network traffic as a temporal
stochastic process and infers hidden regimes such as reconnaissance, intrusion, and data exfiltration
**before attacks fully manifest**.

![Regime Detection](docs/regime_detection.gif)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Attack Regime Modeling](#attack-regime-modeling)
- [Mathematical Formulation](#mathematical-formulation)
- [Model Architecture](#model-architecture)
- [Inference Methods](#inference-methods)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Datasets](#datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [Team](#team)

---

## Problem Statement

Modern networks generate continuous streams of telemetry:
- Packet counts and sizes
- Connection rates and durations
- Protocol usage distributions
- Authentication events
- Flow statistics

Most intrusion detection systems rely on:
- Static threshold rules
- Signature-based matching
- Supervised classifiers on isolated snapshots

These approaches detect attacks **only after clear anomalies appear** — often too late.

This project treats network traffic as a **latent dynamical system** where the true network
condition (normal, scanning, intrusion, exfiltration) is hidden and must be inferred from
noisy, high-dimensional observations over time.

---

## Attack Regime Modeling

Cyber attacks follow multi-stage progressions (aligned with MITRE ATT&CK):

| Stage | Regime Label | Typical Indicators |
|-------|-------------|-------------------|
| 0 | Normal Operation | Stable flows, known protocols |
| 1 | Reconnaissance / Scanning | Port sweeps, DNS lookups, SYN floods |
| 2 | Initial Intrusion | Auth failures, unusual services |
| 3 | Privilege Escalation | Admin logins, policy changes |
| 4 | Lateral Movement | Internal connection spikes |
| 5 | Data Exfiltration | Large outbound flows, unusual destinations |

Rather than detecting isolated events, this project models **attack progression as regime
transitions in a continuous-time stochastic process**.

---

## Mathematical Formulation

### State Dynamics (Regime-Specific)

```
x_t = A_{s_t} * x_{t-1} + w_t,    w_t ~ N(0, Q_{s_t})
```

- `x_t` — hidden network state vector (d-dimensional)
- `A_{s_t}` — regime-specific state transition matrix
- `w_t` — process noise (Gaussian)

### Observation Model

```
y_t = C_{s_t} * x_t + v_t,         v_t ~ N(0, R_{s_t})
```

- `y_t` — observed network telemetry features
- `C_{s_t}` — observation matrix (maps hidden state to observations)
- `v_t` — measurement noise

### Regime Transition (Markov Chain)

```
P(s_t = j | s_{t-1} = i) = π_{ij}
```

- Regime `s_t` follows a discrete Markov chain
- Transition matrix `Π` is learned from data
- Each regime corresponds to an attack stage

### Joint Inference Goal

```
P(x_t, s_t | y_{1:t})  →  regime probabilities + hidden state estimate
```

---

## Model Architecture

```
Raw Network Telemetry (PCAP / Flow logs)
            │
            ▼
    Feature Extraction
    (packet stats, flow metadata, protocol ratios)
            │
            ▼
    Observation Vector  y_t  (normalized, windowed)
            │
            ▼
    ┌───────────────────────────────────┐
    │    Switching State-Space Model    │
    │                                   │
    │  Hidden State  x_t               │
    │  Attack Regime s_t ∈ {0,1,2,3,4,5}│
    │  Transition    Π                  │
    └───────────────┬───────────────────┘
                    │
            Inference Engine
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Kalman Filter   EKF/UKF   Variational
   (linear SSM)  (nonlinear)  Switching KF
                    │
                    ▼
        Regime Probabilities over time
                    │
                    ▼
        Early Attack Stage Detection
```

---

## Inference Methods

| Method | Use Case | Complexity |
|--------|----------|-----------|
| Kalman Filter (KF) | Linear Gaussian SSM baseline | Low |
| Extended Kalman Filter (EKF) | Nonlinear dynamics, local linearization | Medium |
| Unscented Kalman Filter (UKF) | Better nonlinear approximation via sigma points | Medium-High |
| Variational Switching KF (VSKF) | Full switching SSM, regime probabilities | High |

All methods output:
- Posterior state estimate `x_t|t`
- Regime probability vector `P(s_t | y_{1:t})`
- Predicted next observation `y_{t+1|t}`

---

## Repository Structure

```
PARD-SSM/
│
├── README.md                   ← You are here
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    ← Original downloaded datasets (not committed)
│   └── processed/              ← Preprocessed numpy arrays, CSVs
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linear_ssm.py       ← Linear Gaussian SSM definition
│   │   ├── nonlinear_ssm.py    ← Nonlinear SSM (EKF/UKF compatible)
│   │   └── switching_ssm.py    ← Full Switching SSM (core model)
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── kalman_filter.py            ← Standard Kalman Filter
│   │   ├── ekf.py                      ← Extended Kalman Filter
│   │   ├── ukf.py                      ← Unscented Kalman Filter
│   │   └── variational_switching.py    ← Variational Switching KF
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py           ← CICIDS2017 / UNSW-NB15 loaders
│   │   └── feature_engineering.py     ← Feature extraction + normalization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py            ← Regime probability plots
│       └── metrics.py                  ← Evaluation metric computation
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_kalman_filter_demo.ipynb
│   ├── 03_switching_ssm_demo.ipynb
│   └── 04_regime_detection_results.ipynb
│
├── experiments/
│   ├── run_baseline.py         ← Run KF baseline experiment
│   ├── run_switching.py        ← Run full SSSM experiment
│   └── evaluation_metrics.py  ← Aggregate and print results
│
├── tests/
│   ├── test_kalman_filter.py
│   ├── test_switching_ssm.py
│   └── test_data_processing.py
│
└── docs/
    ├── math_derivations.md     ← Full mathematical derivations
    └── dataset_guide.md        ← How to download and prepare datasets
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/prakulhiremath/PARD-in-Network-Traffic-using-Switching-State-Space-Models-.git
cd PARD-SSM
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1 — Download datasets

See `docs/dataset_guide.md` for download links and instructions.
Place raw files in `data/raw/`.

### Step 2 — Preprocess data

```bash
python src/data_processing/dataset_loader.py --dataset cicids2017 --output data/processed/
```

### Step 3 — Run Kalman Filter baseline

```bash
python experiments/run_baseline.py --data data/processed/cicids2017_features.npy
```

### Step 4 — Run full Switching SSM

```bash
python experiments/run_switching.py --data data/processed/cicids2017_features.npy --regimes 4
```

### Step 5 — Visualize regime probabilities

```bash
python src/utils/visualization.py --results experiments/results/switching_output.pkl
```

---

## Datasets

### CICIDS 2017
- Source: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- Contains labeled network flows with attack types
- Attacks: Brute Force, DoS, Infiltration, Port Scan, Botnet, Web Attack

### UNSW-NB15
- Source: [UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- 49 features, 9 attack categories
- Modern attack patterns including reconnaissance and backdoors

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Prediction MSE | Mean squared error of next-step telemetry prediction |
| Log-Likelihood | Model fit score on held-out data |
| Regime Accuracy | Classification accuracy vs ground truth labels |
| Detection Lead Time | How early (in seconds/steps) attack is flagged |
| Confusion Matrix | Per-regime classification breakdown |
| AUC-ROC | Attack vs normal discrimination |

---

## Results

*(Results will be updated as experiments complete)*

| Method | Regime Accuracy | Detection Lead Time | Log-Likelihood |
|--------|----------------|--------------------|-|
| Kalman Filter (baseline) | - | - | - |
| EKF | - | - | - |
| UKF | - | - | - |
| Variational SSSM | - | - | - |

---

## Future Work

- [ ] Real-time streaming inference for live network monitoring
- [ ] Integration with SIEM platforms (Splunk, Elastic)
- [ ] Neural state-space models (deep Kalman filter)
- [ ] Unsupervised discovery of new attack regimes
- [ ] Visualization dashboard (Plotly Dash / Streamlit)
- [ ] Deployment on enterprise network telemetry

---

## Team

| Name | USN |
|------|-----|
| Prakul Sunil Hiremath | 2VX23CS013 |
| Peerahamad Bagawan | 2VX23CS029 |
| Sahil Bekane | 2VX23CS042 |
| Hemanth B. K. | 2VX23CS012 |

**Institution:** VTU affiliated college, Karnataka, India
**Project Type:** Major Project — Computer Science Engineering

---

## Citation

```bibtex
@software{hiremath2026attackregime,
  author    = {Hiremath, Prakul Sunil and Bagawan, Peerahamad and Bekane, Sahil and {Hemanth, B. K.}},
  title     = {Probabilistic Attack Regime Detection in Network Traffic using Switching State-Space Models},
  year      = {2026},
  url       = {https://github.com/prakulhiremath/PARD-in-Network-Traffic-using-Switching-State-Space-Models-},
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds on foundational concepts from:
- Ghahramani & Hinton (1996) — Switching State-Space Models
- Shumway & Stoffer — Time Series Analysis and Its Applications
- MITRE ATT&CK Framework for cyber attack stage taxonomy
- CICIDS and UNSW benchmark dataset creators
