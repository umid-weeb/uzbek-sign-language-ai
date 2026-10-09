# Public Uzbek sign-language dataset

Downloaded from:

<https://github.com/Mrkomiljon/uzbek-sign-language>

Files:

- `data.pickle` — 2,800 MediaPipe hand-landmark samples;
- `README.source.md` — source project description.

The source reports 28 alphabet classes with 100 samples per class. The
pickle stores labels as source folder IDs (`"0"` through `"27"`), not verified
Uzbek letter names. Therefore this data must not be silently merged into the
project's named-label dataset until the source label mapping is confirmed.

This is a static alphabet dataset, not a dynamic word/sentence dataset. It is
kept separately as public Uzbek-domain auxiliary data.

## Reproducible train/test split

The original `dataset.csv` is preserved unchanged. A stratified split was
generated with:

```bash
/usr/local/bin/python3.13 training/split_external_uzbek_dataset.py
```

- `train.csv`: 2,240 samples, 80 per source label;
- `test.csv`: 560 samples, 20 per source label;
- ratio: 80% / 20%;
- seed: 42.

Exact duplicate feature rows are kept in one partition only, preventing
identical samples from appearing in both train and test.

External Random Forest modeli live kamera schema bilan mos kelmagani sabab
olib tashlandi. `dataset.csv`, `train.csv`, `test.csv` va `data.pickle`
saqlanib qoldi; keyingi model aynan RTMLib kamera contractiga mos holda
qayta tayyorlanadi.
