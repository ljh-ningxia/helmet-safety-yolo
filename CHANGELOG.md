# Changelog

All notable changes to this project are documented here.

## [2.0.0] — 2026-11-26

A ground-up rewrite. v1 was a pipeline proof with a synthetic demo detector;
v2 is trained and evaluated on a real, licensed public dataset.

### Added

- **Real data**: Hard Hats Dataset (Roboflow Universe / Hugging Face, CC BY 4.0),
  19,745 real construction images with `hardhat` / `no-hardhat` COCO labels.
  Resumable downloader with SHA256 manifest and pinned upstream revision.
- **Data engineering**: COCO validation (duplicate / orphan / clipped / invalid
  boxes), class-and-density-stratified subset sampling with a stable seed,
  COCO→YOLO conversion, EDA audit (class imbalance, scale buckets, luminance,
  exact-duplicate detection), auto-generated diagnostic charts and a contact
  sheet of real labelled images.
- **Real training**: `scripts/03_train.py` with fixed seed, environment
  fingerprint, exact command capture, Ultralytics metrics and per-epoch logs.
- **Real evaluation**: `scripts/04_evaluate.py` — official mAP on the held-out
  test split, independent CPU latency benchmark (P50/P95), slice metrics by
  luminance and object density, real inference evidence and failure cases.
- **Fair A/B**: four single-variable experiments on the trained model and the
  same real images, reporting both gain and cost.
- **Engineering**: typed dataclass configuration with env overlays, structured
  JSON logging, FastAPI service with graceful 503 when weights are absent,
  web dashboard, unified CLI, 30+ unit tests, GitHub Actions CI, Dockerfile,
  pyproject/requirements, Makefile, ADRs, model card, dataset card, reproduction
  guide and explicit conclusion boundaries.

### Removed

- The synthetic `demo` detector and all `SELFTEST_ONLY` numbers. v2 has no fake
  fallback: inference fails loudly if the trained weights are missing.
- v1's Flask dashboard (replaced by FastAPI).

### Notes

- Training runs on CPU only in this environment; the measured subset size and
  epoch budget are recorded in `configs/data/default.yaml` and
  `experiments/results/train_run.json`. A GPU run can use the full corpus by
  setting the subset values to `null`.

## [1.0.0] — 2026-11-26

Initial pipeline proof: six-stage architecture, three independent innovation
modules, a single-variable A/B harness and a Flask dashboard, all backed by a
synthetic demo detector. Preserved in git history for provenance.
