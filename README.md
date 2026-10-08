# PhysioRoute (PI-DDGN)

Official reproducibility code for PhysioRoute, a physiology-informed directed
delay graph model for multimodal wearable heart-rate forecasting.

The current release implements the leakage-resistant v3 protocol used in the
final manuscript: causal preprocessing, subject-disjoint splits, five-seed
training, input-window-conditioned directed graphs, continuous delayed message
passing, matched neural baselines, graph ablations, PAT controls, robustness
tests, and manuscript figure generation.

## Repository layout

```text
configs/        Dataset, model, and training configurations
src/            Preprocessing, models, baselines, training, and evaluation
scripts/        v3 experiment, audit, aggregation, and plotting entry points
tests/          Unit tests for preprocessing, metrics, models, and delay sampling
results_v3/     Small aggregate JSON files included with the public release
figure_data/    CSV source data for the final manuscript figures
```

Raw datasets, processed arrays, checkpoints, bulk prediction archives, logs,
and personal environment files are intentionally excluded from the public archive.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data

Download the datasets from their original providers and place them at the
following default locations, or edit `raw_dir` in the corresponding YAML file:

```text
data/raw/wesad/
data/raw/ppg_dalia/PPG_FieldStudy/
data/raw/ptt_ppg/physionet.org/files/pulse-transit-time-ppg/1.1.0/
```

The datasets are not redistributed in this archive. Respect each provider's
license and access conditions.

## Leakage-resistant v3 preprocessing

```bash
python -m src.data.preprocess_v3 configs/data/wesad_v3.yaml
python -m src.data.preprocess_v3 configs/data/ppg_dalia_v3.yaml
python -m src.data.preprocess_v3 configs/data/ptt_ppg_v3.yaml
```

The v3 pipeline uses causal filtering and resampling, normalizes from the input
window only, saves record and subject identifiers, and constructs future HR
targets without exposing future information to the model. PTT-PPG additionally
provides future signal-derived PAT for the designated auxiliary diagnostic.

## Train and evaluate

Run one five-seed-compatible job with:

```bash
bash scripts/run_v3_job.sh wesad_v3 picdgn_v3 42 \
  results_v3/canonical/h8/wesad/picdgn_v3/seed42
```

The same runner accepts the baseline model names in `configs/model/`, including
`patchtst`, `lstm`, `gwn`, `deep_ppg`, `mtgnn`, and `itransformer`.

To build the canonical v3 queues:

```bash
python scripts/build_v3_queues.py
bash scripts/run_v3_queue.sh 0 runs/v3/gpu4_picdgn.tsv
```

Queue files are plain TSV and may be distributed across GPUs or machines.

## Aggregate and audit results

```bash
python scripts/aggregate_v3_results.py
python scripts/v3_paired_stats.py
python scripts/audit_v3_runs.py
python scripts/audit_v3_prediction_sanity.py
```

The public archive includes compact aggregate JSON files and the fields needed
to reproduce published figure panels, without redistributing checkpoints or
complete run directories.

## Generate manuscript figures

After reproducing the required aggregate files and PAT prediction artifacts:

```bash
python scripts/make_v3_manuscript_figures.py
```

Figures are exported as PDF, SVG, PNG, and TIFF. The plotting script labels the
proposed method as PhysioRoute; internal configuration keys retain the historical
`picdgn`/`picdgn_v3` names for reproducibility.

## Reproducibility notes

- Canonical experiments use initialization seeds 42--46.
- Standard HR experiments predict a residual from input-period HR.
- Modality-restricted and cross-dataset protocols disable that residual when it
  would be unavailable or privileged.
- Run directories store the resolved configuration and a SHA-256 source manifest.
- PAT values represent a supervised feature-space delay diagnostic and should not
  be interpreted as proof of physical pulse-transit-time recovery or causality.

## Tests

```bash
pytest -q
```

No software license file was present in the source project at packaging time;
add the intended public license before publishing the archive in a repository.
