# Reproducing the experiments

Install the environment described in the [repository README](../README.md).
All commands below run from `code/`. Paths such as `/path/to/vctk` should point
to your downloaded data. Keep `../results/` unchanged and use fresh output
directories for reproduced runs.

## 1. CV17 with independent calibration

The fixed protocol is [`experiments/cv17_protocol.json`](experiments/cv17_protocol.json).
The repository includes the exact train, selection, calibration and test
manifests: 50,000 / 735 / 723 / 1,675 clips. The downloader retrieves the
required files from the pinned
[Common Voice 17.0 mirror](https://huggingface.co/datasets/fsicoli/common_voice_17_0/tree/8262c16bf297c87a9cd88c51997c4758ed7a8ba2).
It streams the English archive shards, so allow tens of GB of network traffic.

```bash
mkdir -p outputs
cp -a ../results/cv17_download_manifests outputs/cv17_audio
python tools/prepare_cv17_release.py --metadata outputs/cv17_audio --out outputs/cv17_audio --fetch
python -c 'from huggingface_hub import snapshot_download; snapshot_download("facebook/wav2vec2-base", revision="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8", allow_patterns=["config.json", "pytorch_model.bin", "preprocessor_config.json"], local_dir="outputs/wav2vec2-base")'
python tools/extract_cv17_features.py --data outputs/cv17_audio --model outputs/wav2vec2-base --out outputs/cv17_features
python tools/run_cv17_parallel.py --bundle outputs/cv17_features --protocol experiments/cv17_protocol.json --out outputs/cv17 --jobs 4
python tools/audit_cv17_release.py --bundle outputs/cv17_features --results outputs/cv17 --out outputs/cv17_audit
```

`--jobs` controls concurrent seed groups on the visible GPU. Use `--jobs 1`
on a memory-constrained device. The full run trains 80 teachers and 310
students. Each condition saves checkpoints, evaluation logits, selection
traces, normalization statistics, and privacy-accountant history. The audit
reconstructs predictions and the paired comparisons used in Tables VIII–XI.

For a single-seed training check, use the same protocol with the serial runner:

```bash
python tools/run_cv17_release.py --bundle outputs/cv17_features --protocol experiments/cv17_protocol.json --out outputs/cv17_seed0 --seeds 0
```

The full audit expects all ten seeds. The existing reference summaries are in
`../results/cv17_release_20260908/audit/`. The pinned NumPy/SciPy versions matter
for exact resampling and accountant replay.

## 2. Small Common Voice archive

Download and extract [Zenodo record 12588635](https://zenodo.org/records/12588635).
Use the archive's Mozilla Public Dataset License 1.0. This experiment uses
6,714 / 1,061 / 459 train/dev/test clips and is separate from CV17.
The following command locates every recorded clip and writes local manifests;
it fails if any required audio is missing.

```bash
python tools/prepare_paper_data.py --dataset small-cv --audio-root /path/to/common_voice_corpus1/extracted --manifests ../results/cv_small_manifests --out outputs/cv_small_data
python tools/prepare_cv_replacement.py --manifest outputs/cv_small_data --audio /path/to/common_voice_corpus1/extracted --model outputs/wav2vec2-base --out outputs/cv_small_features --backend ssl
python tools/prepare_cv_replacement.py --manifest outputs/cv_small_data --audio /path/to/common_voice_corpus1/extracted --model outputs/wav2vec2-base --out outputs/cv_small_features --backend mel
python tools/run_cv_replacement.py --bundle outputs/cv_small_features --out outputs/cv_small --family ssl
python tools/run_cv_replacement.py --bundle outputs/cv_small_features --out outputs/cv_small --family mel
python tools/audit_cv_replacement.py --runs outputs/cv_small --bundle outputs/cv_small_features --out outputs/cv_small_evidence.json
```

Download `outputs/wav2vec2-base` using the pinned command in Section 1.
The two families jointly train 45 teachers and 120 students, with five seeds
and noise multipliers 1, 1.5, and 2. The protocol is
[`experiments/cv_replacement_protocol.json`](experiments/cv_replacement_protocol.json).
Historical SSL and Mel runs used NumPy 2.4.6 and 2.5.3, respectively; the
included `mel_resampling_reference.json` records the latter's resampling
values. For a new run, audit with the same environment used to train it.

The additional Mel-BS condition in Supplement III-C uses ten teachers at
noise multipliers 1 and 1.5. Its recorded configurations preserve the distinct
class-count convention and feature path:

```bash
python tools/replay_recorded.py --source ../results/common_voice_paired_20260907 --data-root outputs/cv_small_data --out outputs/cv_mel_bs
python tools/run_ctkd_global.py --source outputs/cv_mel_bs --pattern '*hard.json' --out outputs/cv_mel_bs_ctkd
```

## 3. VCTK and heterogeneous transfer

Obtain [VCTK 0.92](https://doi.org/10.7488/ds/2645) and extract the
`wav48_silence_trimmed` audio. The included manifests select 720 utterances
from English, Irish, Scottish, and US speakers. The 480-row training split
is divided into 360 private and 120 auxiliary records for each seed;
development and test each contain 120 records.

```bash
python tools/prepare_paper_data.py --dataset vctk --audio-root /path/to/vctk --manifests ../results/vctk_manifest_20260907 --out outputs/vctk_data
python tools/replay_recorded.py --source ../results/strict_bs_pair --data-root outputs/vctk_data --out outputs/vctk
python tools/run_development_grid.py --source outputs/vctk --out outputs/vctk_dev_grid
python tools/run_ctkd_global.py --source outputs/vctk --pattern '*tagstrictpair_bs_s*_hard.json' --out outputs/vctk_ctkd --tag ctkd_global_20260907
```

The first replay covers the paired branches, HAKD weight variants, student
selectors, and all 30 health-weight runs. It preserves the recorded
experimental parameters while relocating file paths. It trains each missing
teacher once and shares that checkpoint across its branches. Add `--dry-run`
to inspect configurations without training, or `--seeds 0` for a short run.
The development grid retrains with 30, 60, and 120 development examples and
evaluates all four threshold pairs.

Matched non-private teacher and PATE controls use the same VCTK features:

```bash
python tools/run_pate_matched.py --source outputs/vctk --bundle outputs/vctk_bundle --prepare
python tools/run_vctk_teacher_controls.py --bundle outputs/vctk_bundle --out outputs/vctk_nonprivate
python tools/run_pate_matched.py --bundle outputs/vctk_bundle --out outputs/vctk_pate
python tools/audit_pate_matched.py --bundle outputs/vctk_bundle --results outputs/vctk_pate --out outputs/vctk_pate_check.json
```

The PATE comparison trains three teachers on stable disjoint shards, privatizes
120 auxiliary count queries, and trains Hard, PATE, and PATE-plus-labels
students. Its Gaussian RDP bound matches the reported single-teacher PRV bound
at epsilon 6.0829038771 and delta 1e-5; the accounting mechanisms differ.

The weak Mel-CE experiment uses the same audio manifests:

```bash
python tools/replay_recorded.py --source ../results/vctk_ce_grid --data-root outputs/vctk_data --out outputs/vctk_weak_ce
```

For the frozen HuBERT teacher and mel-CNN student, obtain
[`facebook/hubert-base-ls960`](https://huggingface.co/facebook/hubert-base-ls960)
and supply its local model directory:

```bash
python tools/replay_recorded.py --source ../results/hubert_arch_hetero --data-root outputs/vctk_data --ssl-model /path/to/hubert-base --out outputs/vctk_hubert
python tools/run_ctkd_global.py --source outputs/vctk_hubert --pattern '*hard.json' --out outputs/vctk_hubert_ctkd
```

Tables XIII and XVIII distinguish the published `ctkd_global` method from the
older `ctkd` linear-temperature probe. The implementation-level comparison
uses [CTKD commit 56112892](https://github.com/zhengli97/CTKD/tree/56112892d5aca069bd56bb057feee9f7ff9e4141):

```bash
git clone https://github.com/zhengli97/CTKD.git outputs/CTKD
git -C outputs/CTKD switch --detach 56112892d5aca069bd56bb057feee9f7ff9e4141
python tools/verify_ctkd_reference.py --reference outputs/CTKD --out outputs/ctkd_reference_check.json
```

## 4. IEMOCAP feature classification

Download these three files from
[Zenodo record 17803295](https://doi.org/10.5281/zenodo.17803295), the CC BY 4.0
WavLM-large feature release by D. Yohanes and A. Chowanda:

- `wavlm-large_train_6s_stacked_gauss_speed_reverb.pkl`
- `wavlm-large_val_6s_original.pkl`
- `wavlm-large_test_6s_original.pkl`

The source files are pandas pickles; load only the publisher's trusted files.
The USC raw IEMOCAP corpus has separate access terms.

```bash
python tools/prepare_iemocap_precomputed.py --train /path/to/wavlm-large_train_6s_stacked_gauss_speed_reverb.pkl --dev /path/to/wavlm-large_val_6s_original.pkl --test /path/to/wavlm-large_test_6s_original.pkl --out outputs/iemocap_pooled
python tools/prepare_iemocap_disjoint.py --pooled outputs/iemocap_pooled --out outputs/iemocap_data
python tools/run_iemocap_precomputed.py --data_dir outputs/iemocap_data --out_dir outputs/iemocap --seeds 0 1 2 3 4
```

Both preparation steps are required. The second pools identified variants per
utterance, derives speakers from the utterance suffix, and assigns sessions
1–3 / session 4 / session 5 female / session 5 male to private / auxiliary /
development / test. The resulting sizes are 3,259 / 1,031 / 590 / 651.
The five runs measure optimization variation on this fixed speaker split.

## 5. Release timing

After the two small Common Voice families in Section 2 have completed:

```bash
python tools/benchmark_release_matched.py --bundle outputs/cv_small_features --teachers outputs/cv_small --out outputs/release_timing
```

The benchmark trains every student candidate for 15 epochs. It runs two model
families, three seeds, three repeats, and six policies, with CUDA synchronization
and resident features. S-BAcc trains all three candidate students. Common
teacher training and feature extraction are excluded. These measurements
describe the small-archive protocol, not the 80-epoch CV17 experiment.

## 6. Outputs and checks

Training produces JSON metrics and configuration records, teacher/student
checkpoints, and evaluation predictions. The CV17 and small-archive audits
recompute metrics and replay the saved accountant histories. The PATE audit
checks noisy queries and matched student outputs. Generic replay additionally
saves evaluation predictions for its reproduced branches.

```bash
python -m pytest -q tests
```

See the [root README](../README.md#recreate-the-figures-and-tables) for rendering
the recorded results without retraining. Teacher-health intervals concern the
specified Gaussian region model; they are not student-utility guarantees.
Seed intervals summarize repeated runs on the fixed datasets.
