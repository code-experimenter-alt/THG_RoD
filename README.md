
## Overview

This repository implements a release-oriented framework for privacy-preserving speech classification under class imbalance.

We train a differentially private teacher on sensitive speech data, query it once offline on permitted auxiliary data, and train a released student using hard-label supervision or distillation-based supervision. By differential privacy post-processing, the released student inherits the teacher-side privacy guarantee with respect to the private teacher-training data.

Our study focuses on a key release-stage question:

> When does a private teacher remain healthy enough to guide the released student?

The main findings of this work are:

- Teacher health matters more than teacher accuracy alone for released-student utility.
- Under class imbalance, a differentially private teacher may retain moderate accuracy while collapsing toward majority-class prediction.
- In the main frozen SSL-head teacher setting, Balanced Softmax yields healthier private teachers than standard cross-entropy.
- Teacher-Conditioned Release Distillation, or TCRD, acts as a conservative release policy that preserves knowledge distillation when the teacher is healthy and avoids negative transfer when teacher supervision is unreliable.

---

## Release Protocol

We use the following release-oriented private learning protocol:

1. Train a differentially private teacher on private data `D_priv`.
2. Query the teacher once offline on permitted auxiliary data `D_aux`.
3. Train a student on `D_aux` using one of the following supervision modes:
   - hard labels
   - standard knowledge distillation
   - health-aware knowledge distillation
   - teacher-conditioned routing
4. Release only the student.

The teacher is not released and is not deployed as a persistent query interface.

---

## Privacy Scope

The reported privacy guarantee applies only to the private teacher-training set `D_priv`.

- The teacher is trained with example-level DP-SGD.
- The released student inherits the same guarantee with respect to `D_priv` by post-processing.
- No differential privacy protection is claimed for:
  - `D_aux`
  - `D_dev`
  - `D_test`

If auxiliary or development data are sensitive, additional privacy protection or accounting is required.

---

## Repository Structure

```text
THG-RoD/
├── README.md
├── requirements.txt
├── .gitignore
├── experiments/
│   ├── __init__.py
│   └── presets.py
├── tools/
│   ├── aggregate_release_json.py
│   └── compare_hard_vs_kd.py
└── src/
    ├── __init__.py
    ├── main.py
    ├── dataset.py
    ├── features.py
    ├── models.py
    ├── losses.py
    ├── metrics.py
    ├── routing.py
    ├── train_teacher.py
    └── train_student.py
```

---

## Environment

Recommended environment:

- Python 3.9+
- PyTorch
- torchaudio
- Opacus
- Transformers

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dependencies

Main packages used in this repository:

- `torch`
- `torchaudio`
- `torchvision`
- `opacus`
- `transformers`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `pyyaml`

---

## Data Preparation

### Supported Datasets

This repository is primarily organized around the Common Voice main benchmark used in the paper.

The paper also reports results on VCTK and IEMOCAP. At the current stage, the public code is centered on the Common Voice release pipeline and modular components for:

- DP teacher training
- offline teacher supervision
- student release training
- health-aware distillation
- release-result aggregation

### Common Voice Example Layout

```text
data/common_voice/
├── cv-valid-train.csv
├── cv-valid-dev.csv
├── cv-valid-test.csv
├── cv-valid-train/
├── cv-valid-dev/
└── cv-valid-test/
```

The CSV files should include:

- label column: `accent`
- grouping column if available: `client_id`
- audio path column: `filename` or `path`

### Label Setting for Common Voice

We use a 3-way accent classification setup:

- `US`
- `England`
- `Indian`

Internally, the code canonicalizes common variants of these labels into:

- `us`
- `england`
- `indian`

### Common Voice Split Protocol

For Common Voice, we follow the official train/dev/test split and further divide the official training set into:

- `D_priv`: private teacher-training subset
- `D_aux`: permitted auxiliary subset

When available, the split uses grouped splitting by `client_id`.

This design is intended to reduce contamination between private training and auxiliary supervision.

---

## Main Methods Implemented

### 1. DP Teacher Training

The teacher is trained on private data using DP-SGD with Opacus.

Supported teacher objectives:

- `ce`
- `balanced_softmax`
- `logit_adjust`

### 2. Offline Teacher Supervision

The trained teacher is queried once offline on auxiliary data `D_aux`, and its logits are cached for student training.

### 3. Released Student Training

The released student can be trained with one of the following modes:

- `hard`: hard-label training only
- `kd`: standard knowledge distillation
- `hakd`: health-aware knowledge distillation
- `tcrd`: teacher-conditioned routing among hard-label training, HAKD, and KD

---

## Teacher Health

The code computes collapse-aware teacher diagnostics on the development set, including:

- balanced accuracy
- macro-F1
- majority prediction rate
- per-class recall

Global teacher health is defined from development-set teacher behavior and used for routing in TCRD.

---

## Main Presets

Experiment presets are defined in:

```text
experiments/presets.py
```

Main presets include:

- `CV_SSL_BS_RELEASE`
- `CV_MEL_CE_WEAK_RELEASE`
- `CV_MEL_BS_WEAK_RELEASE`
- `G0a_NonDP_cnn_vanilla`
- `G0b_SSL_linear_nonDP`
- `G1a_DPdirect_cnn_vanilla`
- `G1b_DPdirect_ssl_linear`

Minimal preset set:

- `CV_SSL_BS_RELEASE`
- `CV_MEL_CE_WEAK_RELEASE`

---

## Running Experiments

### 1. Main Common Voice Healthy-Teacher Release Pipeline

```bash
python -m src.main \
  --data_root data/common_voice \
  --exp_id CV_SSL_BS_RELEASE \
  --seeds 0 1 2 3 4 \
  --sigmas 1.0 1.5 2.0 \
  --split_mode official \
  --out_dir outputs/common_voice \
  --cache_dir outputs/cache
```

### 2. Weak-Teacher Diagnostic Pipeline

```bash
python -m src.main \
  --data_root data/common_voice \
  --exp_id CV_MEL_CE_WEAK_RELEASE \
  --seeds 0 \
  --sigmas 1.0 1.5 \
  --split_mode official \
  --out_dir outputs/common_voice \
  --cache_dir outputs/cache
```

### 3. Run the Minimal Preset Set

```bash
python -m src.main \
  --data_root data/common_voice \
  --run_all_minimal \
  --out_dir outputs/common_voice \
  --cache_dir outputs/cache
```

---

## Reusing a Cached Teacher

Teacher checkpoints and cached auxiliary logits are reused automatically when possible.

### Step 1: Train and Save the Teacher

```bash
python -m src.main \
  --data_root data/common_voice \
  --exp_id CV_SSL_BS_RELEASE \
  --seeds 0 \
  --sigmas 1.0 \
  --train_teacher 1
```

### Step 2: Reuse the Cached Teacher

```bash
python -m src.main \
  --data_root data/common_voice \
  --exp_id CV_SSL_BS_RELEASE \
  --seeds 0 \
  --sigmas 1.0 \
  --train_teacher 0 \
  --student_mode kd
```

---

## Useful Command-Line Arguments

### Core Arguments

- `--data_root`: dataset root directory
- `--exp_id`: experiment preset name
- `--run_all_minimal`: run the minimal preset list
- `--out_dir`: directory for JSON outputs and checkpoints
- `--cache_dir`: directory for cached features

### Split Arguments

- `--split_mode official|custom`
- `--group_col`

### Teacher Arguments

- `--train_teacher 0|1`
- `--dp_teacher 0|1`
- `--loss_name ce|balanced_softmax|logit_adjust`
- `--max_grad_norm`
- `--delta`
- `--lr_teacher`
- `--epochs_teacher`

### Student and Release Arguments

- `--student_mode hard|kd|hakd|tcrd`
- `--model_student linear_head|cnn_mel`
- `--student_backend ssl|mel`
- `--kd_alpha`
- `--kd_temperature`
- `--lr_student`
- `--epochs_student`

### Feature Arguments

- `--n_mels`
- `--max_time`
- `--use_dsaf 0|1`
- `--eta0`
- `--target_sr`

### TCRD and Health Arguments

- `--health_power`
- `--health_floor`
- `--tcrd_tau_low`
- `--tcrd_tau_high`

---

## Output Files

Each run produces a JSON result file containing:

- experiment configuration
- teacher metrics
- student metrics
- privacy information
- teacher health
- routing information
- teacher cache ID

Example output file:

```text
outputs/common_voice/CV_SSL_BS_RELEASE_seed0_sigma1.json
```

Teacher checkpoint example:

```text
outputs/common_voice/<teacher_cache_id>_teacher.pt
```

Cached teacher logits example:

```text
outputs/common_voice/<teacher_cache_id>_soft_logits_aux.npy
```

---

## Result Aggregation

### Aggregate All JSON Outputs into CSV Summaries

```bash
python tools/aggregate_release_json.py \
  --input_dir outputs/common_voice \
  --out_dir outputs/common_voice/summary
```

This produces:

- `all_runs_flat.csv`
- `core_runs.csv`
- `summary_mean_std.csv`
- `mechanism_table.csv`
- `health_vs_gain.csv`

### Compare Paired Hard-Label and KD Runs

```bash
python tools/compare_hard_vs_kd.py \
  --input_dir outputs/common_voice \
  --hard_tag hardOnly_sweepSigmaGrid_v1_fixedTeacher \
  --kd_tag kd_sweepSigmaGrid_v1_a0.5_T8_fixedTeacher \
  --out_csv outputs/common_voice/hard_vs_kd_summary.csv
```

---

## Default Experimental Settings

### Audio

- sampling rate: `16 kHz`

### Mel Frontend

- `40` mel bins
- `100` frames

### SSL Backend

- frozen `facebook/wav2vec2-base`
- mean pooling
- maximum input length: `12 s`

### Teacher Training

- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-4`
- batch size: `128`
- epochs: `8`

### Differential Privacy

- clipping norm: `C = 8.0` in the main release setting
- default noise multiplier example: `sigma = 1.0`
- `delta = 1e-5`

### Student Training

- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-4`
- epochs: `15`

### Knowledge Distillation

- temperature: `T = 2`
- weight: `alpha = 0.7`

---

## Reproducibility Notes

- The code fixes random seeds for Python, NumPy, and PyTorch.
- Audio is automatically resampled to `16 kHz`.
- The SSL backend uses frozen representations by default.
- Audio paths are automatically resolved across several common folder layouts.
- Feature caches and teacher checkpoints are reused when available.
- Common Voice grouping uses `client_id` when available to reduce speaker leakage across `D_priv` and `D_aux`.

---

## Minimal Test Workflow

First confirm the repository structure:

```text
THG-RoD/
├── README.md
├── requirements.txt
├── .gitignore
├── experiments/
│   ├── __init__.py
│   └── presets.py
├── tools/
│   ├── aggregate_release_json.py
│   └── compare_hard_vs_kd.py
└── src/
    ├── __init__.py
    ├── main.py
    ├── dataset.py
    ├── features.py
    ├── models.py
    ├── losses.py
    ├── metrics.py
    ├── routing.py
    ├── train_teacher.py
    └── train_student.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a minimal experiment:

```bash
python -m src.main \
  --data_root data/common_voice \
  --exp_id CV_SSL_BS_RELEASE \
  --seeds 0 \
  --sigmas 1.0 \
  --out_dir outputs/common_voice \
  --cache_dir outputs/cache
```

Aggregate results:

```bash
python tools/aggregate_release_json.py \
  --input_dir outputs/common_voice \
  --out_dir outputs/common_voice/summary
```

---

## Contact

For questions regarding the code or paper, please contact the authors through the affiliations listed in the manuscript.
