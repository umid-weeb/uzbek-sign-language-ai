# Imported sign2text artifacts

Source repository: <https://github.com/uzibytes/sign2text>

Downloaded artifacts:

- `model.p`
- `data.pickle`

The source model returns 33 string class IDs (`"0"` through `"32"`). Its
published data contains 2,849 rows with 42 features, and the bundled model
reproduces 100% on those same rows. This is an in-sample compatibility check,
not an independent or live-camera accuracy result.

The project exposes the model only through:

```bash
/usr/local/bin/python3.13 pose/webcam_predict.py --model sign2text
```

That mode converts RTMLib pixel coordinates to normalized coordinates and
applies the source preprocessing `(x - min_x, y - min_y)`. It remains separate
from the Uzbek model because the source labels and sign conventions are not
verified as Uzbek.
