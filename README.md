# VTC Analysis via a CPST Lens

Code and results for the worked example in the manuscript
*"Beyond Predictive Accuracy: A CPST-Based Indicator Architecture for Assessing Environmental AI"*
(Gao, Ayana, Chen, Shi, Ding, Ning; submitted to *Ecological Indicators*).

The manuscript proposes **validated-threshold coverage (VTC)** as a Level 2 indicator of the
Cyber-Thinking relation in the Cyber-Physical-Social-Thinking (CPST) framework: it measures whether the
species detections that enter an AI-derived biodiversity indicator are warranted at the sites assessed,
not how accurate the classifier is in general. This repository computes VTC once, for BirdNET, on a
public, fully annotated soundscape dataset.

## Result (p\* = 0.95, m = 50)

| Quantity | Value |
|---|---|
| Recordings | 33 × 1 h, Sierra Nevada, California (2018) |
| Detections at score ≥ 0.10 (3-s windows) | 88,736, of which 39.8 % are true positives |
| Species detected / annotated | 53 / 56 |
| Species with a validated threshold | 19 |
| **VTC** (unweighted; detection-weighted) | **0.36**; 0.56 |
| Failed: fewer than m verified detections | 20 species |
| Failed: p\* not reached at any score | 13 species |
| Failed: p\* reached only by extrapolation | 1 species |
| Validated species: median precision above τ | 0.99 |
| Validated species: median share of annotated windows detected above τ | 0.18 |
| VTC at p\* = 0.90 / 0.95 / 0.99 (m = 50) | 0.51 / 0.36 / 0.06 |
| VTC at m = 20 / 50 / 100 (p\* = 0.95) | 0.40 / 0.36 / 0.30 |

Read through CPST, the binding constraint lies in the Cyber-Thinking relation, with two failing
dimensions: **procedural** for the 20 species short of verified detections (more verification would help)
and **semantic** for the 13 species whose scores do not separate true from false positives (more
verification alone would not).

This is an illustration of the indicator, not an evaluation of BirdNET. The annotator ignored calls that
were too faint or unidentifiable, so some detections counted as false positives may be real calls;
precision, and therefore VTC, is probably underestimated here.

## Data

Kahl, S., Wood, C.M., Chaon, P., Peery, M.Z., Klinck, H. (2022). *A collection of fully-annotated
soundscape recordings from the Western United States.* Zenodo, version 1.
https://doi.org/10.5281/zenodo.7050014 (CC BY 4.0)

The audio and annotations are **not** redistributed here. Download from Zenodo:

- `soundscape_data.zip` (3.2 GB; MD5 `2874bdb56f18002a0e5b6fe2402b3c0b`) → unzip into `audio/`
- `annotations.csv`
- `species.csv`

## Reproduce

Python 3.10–3.14.

    python -m pip install -r requirements.txt
    python run_birdnet.py audio species.csv detections.csv
    python compute_vtc.py detections.csv annotations.csv species.csv results

`run_birdnet.py` runs BirdNET model version 2.4 (LiteRT runtime) on consecutive 3-s windows,
restricted to the 56 annotated species, keeping detections with confidence ≥ 0.10. It processes one
recording at a time and resumes if interrupted. Five recordings (SNE_006, 008, 014, 023, 032) cannot be
read by libsndfile ("flac decoder lost sync"), even from the MD5-verified zip; they are decoded with
FFmpeg through PyAV. On a recording that both decoders can read, they return identical samples.
See `results/decoder_log.txt`.

`compute_vtc.py` labels a detection as a true positive when an annotation of the same species overlaps
its window, fits a logistic regression of true-positive status on confidence score for each species, and
takes τ as the lowest score at which the fitted probability reaches p\* (following Wood & Kahl 2024,
*J. Ornithol.* 165:777–782, https://doi.org/10.1007/s10336-024-02144-5). A species is validated if it has
at least m verified detections and τ lies within the observed scores. The dataset is one stratum.

## Contents

| File | Description |
|---|---|
| `run_birdnet.py` | Step 1: BirdNET inference |
| `compute_vtc.py` | Step 2: VTC, per-species thresholds, sensitivity grid |
| `results/detections.csv` | BirdNET detections used in the manuscript (88,736 rows) |
| `results/vtc_summary.json` | Headline VTC and sensitivity grid |
| `results/vtc_species_table.csv` | Per-species status, threshold, precision, and recall |
| `results/decoder_log.txt` | Recordings decoded with FFmpeg |

## Software versions used

birdnet 1.4.0 (BirdNET model 2.4), ai-edge-litert 2.3.0, PyAV 19.0.2, soundfile 0.14.0,
statsmodels 0.15.0, pandas 3.0.5, NumPy 2.5.3. The analysis was run on Windows with Python 3.14
and reproduced on Linux with Python 3.13 from the same detections file.

## Note on AI assistance

The scripts were drafted with the assistance of Claude (Anthropic). The authors ran the code, checked its
output, and are responsible for the analysis.
