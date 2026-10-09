# Public Uzbek dynamic sign-language checkpoint

Source repository:

<https://github.com/akkomron/uzslr-isolated-dynamic>

Downloaded assets:

- `best_model.pth` — pretrained 50-class Uzbek isolated-sign checkpoint;
- `inference01_config.py` — source labels and runtime settings;
- `inference02_preprocess.py` — source MediaPipe Holistic preprocessing;
- `inference03_model.py` — source CNN/Transformer architecture;
- `inference04_main.py` — source webcam inference;
- `README.md` — source documentation.

The source README reports approximately 92% validation accuracy and 87% test
accuracy on its own Uzbek dataset. These metrics are not a result measured on
this project's dataset and must not be presented as this project's benchmark.

The source dataset itself is not included in the repository. The checkpoint
expects 32-frame MediaPipe Holistic sequences with 708 preprocessed channels
and predicts 50 Uzbek isolated signs. It is kept separate from the project's
42-feature static model and 378-feature prototype.
