# Sign Language Dataset Hub snapshot

Source repository:

<https://github.com/rudra496/SignLanguage-Dataset-Hub>

The repository was cloned on 2026-10-09 and contains `datasets_catalog.csv`
with 66 catalog rows covering multiple sign languages and modalities. The
catalog does not contain an Uzbek entry. It is a verified catalog, not one
single downloadable dataset.

## Downloaded data

The repository-included CC BY 4.0 BdSL sensor demo was downloaded without
changing it:

- `bdsl-sensor-glove/train/data_train.json`: 3,528 recordings;
- `bdsl-sensor-glove/val/data_val.json`: 648 recordings;
- `bdsl-sensor-glove/test/data_test.json`: 648 recordings.

This is Bangla Sign Language sensor-glove data with 36 gesture classes and 11
sensor channels. It is not Uzbek hand-camera data and must not be merged into
the Uzbek alphabet classifier.

## Why the other datasets were not bulk-downloaded

The catalog links to independent sources. Several require Kaggle credentials,
registration or research approval; others are 15 GB to 1.2 TB, and licenses
include research-only, non-commercial, BBC and custom terms. The hub's own
downloader only provides direct URLs for a small subset and cannot legally or
technically download every catalog entry automatically.

The catalog, source links, attribution and license information remain locally
available for targeted future downloads.
