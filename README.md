# VTC Analysis via a CPST Lens

Code and results for the worked example in the manuscript
*"Beyond Predictive Accuracy: A CPST-Based Indicator Architecture for Assessing Environmental AI"*
(Gao, Ayana, Chen, Shi, Ding, Ning; submitted to *Ecological Indicators*).

The manuscript proposes **validated-threshold coverage (VTC)** as a gate from Level 1 (AI-system
performance) to Level 2 (environmental assessment) in the Cyber-Physical-Social-Thinking (CPST) framework.
VTC is the share of species whose classifier threshold has been validated locally, so their detections may be
used as evidence in occupancy or richness indicators. It tests the Cyber-Thinking relation: whether model
outputs warrant the conclusions drawn from them at the sites assessed. This repository computes VTC for
two BirdNET versions on a public, fully annotated soundscape dataset, checks it on held-out recordings, and
shows what happens to the indicator when the model changes.

## Results (p\* = 0.95, m = 50, score floor 0.10)

| Quantity | BirdNET 2.4 | BirdNET+ 3.0 (developer preview 3.1) |
|---|---|---|
| Detections; share that are true positives | 88,736; 39.8 % | 38,089; 69.5 % |
| Species detected / annotated | 53 / 56 | 56 / 56 |
| Species validated in sample | 19 | 15 |
| **VTC in sample** (unweighted; detection-weighted) | **0.36**; 0.56 | **0.27**; 0.62 |
| Held out (5-fold by recording): thresholds meeting p\*; below; not checkable | 12; 2; 5 | 9; 4; 2 |
| **VTC held out** | **0.23** | **0.16** |
| Bootstrap 95 % interval for VTC (1,000 draws by recording) | 0.27–0.43 | 0.20–0.32 |
| Validated species stable in ≥ 90 % of draws | 6 of 19 | 9 of 15 |
| VTC under the lower-confidence-bound rule (species) | 0.13 (7) | 0.20 (11) |
| Validated species: median precision above τ; median share of annotated windows above τ | 0.99; 0.18 | 0.98; 0.18 |
| VTC at p\* = 0.90 / 0.95 / 0.99 (m = 50) | 0.51 / 0.36 / 0.06 | 0.39 / 0.27 / 0.05 |
| VTC at m = 20 / 50 / 100 (p\* = 0.95) | 0.40 / 0.36 / 0.30 | 0.29 / 0.27 / 0.25 |

**Model update.** The two versions validate 11 species in common; 8 are validated only with 2.4 and 4 only with
3.0 (including Golden-crowned Kinglet, the most frequently detected species, which fails with 2.4). Thresholds
of the shared species move by a median of 0.11. A biodiversity indicator whose time series spanned the update
would change its species list and thresholds without any change in the birds, which is why the manuscript's
revision rule requires re-fitting thresholds, recomputing VTC, and re-baselining the series after a model change.

**Reading through CPST.** The binding constraint lies in the Cyber-Thinking relation, with two failing
dimensions: **procedural** for species short of verified detections (more verification would help) and
**semantic** for species whose scores do not separate true from false positives (more verification alone would not).

This is an illustration of the indicator, not an evaluation of BirdNET. The annotator ignored calls that were too
faint or unidentifiable, so some detections counted as false positives may be real calls; precision, and therefore
VTC, is probably underestimated here. VTC covers false positives only; detection probability must be estimated
separately.

## Data

Kahl, S., Wood, C.M., Chaon, P., Peery, M.Z., Klinck, H. (2022). *A collection of fully-annotated
soundscape recordings from the Western United States.* Zenodo, version 1.
https://doi.org/10.5281/zenodo.7050014 (CC BY 4.0)

The audio and annotations are **not** redistributed here. Download from Zenodo:

- `soundscape_data.zip` (3.2 GB; MD5 `2874bdb56f18002a0e5b6fe2402b3c0b`) → unzip into `audio/`
- `annotations.csv`
- `species.csv`

BirdNET+ V3.0 developer preview 3.1: https://doi.org/10.5281/zenodo.20703646 (CC BY-SA 4.0), downloaded
automatically by the `birdnet` package on first use.

## Reproduce

Python 3.10–3.14.

    python -m pip install -r requirements.txt

    # BirdNET 2.4
    python run_birdnet.py audio species.csv detections.csv
    python compute_vtc.py detections.csv annotations.csv species.csv results
    python validate_vtc.py detections.csv annotations.csv species.csv heldout

    # BirdNET+ 3.0 (developer preview)
    python run_birdnet.py audio species.csv detections_v3.csv 3.0
    python compute_vtc.py detections_v3.csv annotations.csv species.csv results_v3
    python validate_vtc.py detections_v3.csv annotations.csv species.csv heldout_v3

    # Revalidation after the model change
    python compare_versions.py results v2.4 results_v3 v3.0 comparison

Each BirdNET run takes about 1–1.5 minutes per recording hour on a laptop. `run_birdnet.py` processes one
recording at a time and resumes if interrupted. Five recordings (SNE_006, 008, 014, 023, 032) cannot be read by
libsndfile ("flac decoder lost sync"), even from the MD5-verified zip; they are decoded with FFmpeg through PyAV.
On a recording that both decoders can read, they return identical samples.

## Method

- **True positive:** an annotation of the same species overlaps the detection's 3-s window.
- **Threshold τ:** per species, a logistic regression of true-positive status on confidence score; τ is the
  lowest score at which the fitted probability reaches p\* (following Wood & Kahl 2024, *J. Ornithol.*
  165:777–782, https://doi.org/10.1007/s10336-024-02144-5). A species is validated if it has at least m verified
  detections and τ lies within the observed scores.
- **VTC** = validated species / species detected. The dataset is one stratum (one region, one recorder type).
- **Held out** (`validate_vtc.py`): five-fold cross-validation grouped by recording; a threshold counts if its
  pooled held-out precision reaches p\*. Cluster bootstrap by recording (1,000 draws, seed 20261117) for
  intervals on VTC and τ. Lower-bound rule: the lower 95 % confidence bound of the fitted probability must reach p\*.
- **Revalidation** (`compare_versions.py`): same recordings, annotations, and settings for both versions.

## Contents

| Path | Description |
|---|---|
| `run_birdnet.py` | BirdNET inference (model 2.4 or 3.0) |
| `compute_vtc.py` | VTC, per-species thresholds, sensitivity grid |
| `validate_vtc.py` | Held-out cross-validation, bootstrap, lower-bound rule |
| `compare_versions.py` | Comparison of two model versions |
| `results/` | BirdNET 2.4: detections, VTC summary, per-species table, decoder log |
| `results_v3/` | BirdNET+ 3.0: the same |
| `heldout/`, `heldout_v3/` | Held-out and bootstrap results for each version |
| `comparison/` | Version comparison (summary and per species) |

## Software versions used

birdnet 1.4.0 (BirdNET model 2.4 via ai-edge-litert 2.3.0; BirdNET+ 3.0 preview 3.1 via onnxruntime 1.31.1),
PyAV 19.0.2, soundfile 0.14.0, statsmodels 0.15.0, pandas 3.0.5, NumPy 2.5. All analyses were run on Windows
with Python 3.14 and reproduced on Linux with Python 3.13 with identical results.

## Note on AI assistance

The scripts were drafted with the assistance of Claude (Anthropic). The authors ran the code, checked its
output, and are responsible for the analysis.
