# Third-party notices

This project bundles or downloads the following third-party assets. They are
NOT covered by this repository's MIT code license.

## 1. Hard Hats Dataset (training / evaluation images and labels)

- Source: Roboflow Universe, "Hard Hats Dataset", 2022
  https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5/dataset/2
- Mirror used by this project: https://huggingface.co/datasets/keremberke/hard-hat-detection
- License: **CC BY 4.0** (declared both in the dataset card and inside every
  `_annotations.coco.json` under `licenses[].url`)
- Obligation: attribution is required. This NOTICE plus `docs/DATASET.md`
  provide it. If you redistribute images or derived visualisations, keep this
  attribution.
- Suggested citation:

  ```bibtex
  @misc{hard-hats-fhbh5_dataset,
    title = {Hard Hats Dataset},
    type = {Open Source Dataset},
    author = {Roboflow Universe Projects},
    howpublished = {\url{https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5}},
    publisher = {Roboflow},
    year = {2022}, month = {dec}
  }
  ```

## 2. Ultralytics YOLOv8n pretrained weights

- File: `models/yolov8n.pt` (SHA256 recorded in `models/weights_manifest.json`)
- Source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
- License: **AGPL-3.0** for Ultralytics software/weights. Using the weights to
  produce a trained model does not relicense this repository's own Python code,
  but any distribution of the weights or an AGPL-covered derivative must comply
  with AGPL-3.0. For commercial deployment, obtain an Ultralytics Enterprise
  licence or train from a permissively licensed backbone.
- The weight file is **not committed** to this repository (see `.gitignore`).

## 3. Python dependencies

`ultralytics`, `torch`, `opencv`, `fastapi`, `matplotlib`, etc. retain their own
licences; see each package's metadata. `requirements-lock.txt` records the exact
versions used for the reported measurements.

## Own code

Original code in `src/`, `scripts/`, `tests/` and `web/` is MIT-licensed, see
`LICENSE`.
