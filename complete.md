# Bajarilgan ishlar hisoboti

## Sana

2026-10-09

## Qisqa xulosa

Loyiha static harf klassifikatsiyasidan dynamic imo-ishora recognition
pipeline'iga o'tkazish uchun tayyorlandi. Mavjud `dataset.csv` va static
model o'chirilmadi. Yangi pipeline ikki qo'l landmarklarini vaqt bo'yicha
sequence sifatida yig'adi va Temporal Transformer modelida o'qitadi.

Muhim cheklov: dynamic model uchun hali real sequence dataset yig'ilmagan.
Shuning uchun dynamic modelning accuracy foizi hozircha hisoblanmagan va
uni tayyor natija deb ko'rsatish noto'g'ri bo'ladi.

---

## 1. Mavjud static model bo'yicha natijalar

### Dataset

- Dataset: `data/dataset.csv`
- Umumiy namunalar: 6547 ta
- Test namunalar: 1311 ta
- Klasslar: 29 ta
- Feature: 42 ta (`21 landmark x 2 koordinata`)

### O'lchangan natijalar

| Model | Test accuracy | Holat |
|---|---:|---|
| Random Forest | **93.29%** | Ishlaydigan static baseline |
| MLP | **91.23%** | Ishlaydigan static deep-learning baseline |

Random Forest MLP'dan:

```text
93.29% - 91.23% = 2.06 foiz punkt yuqori
```

Bu `2.06%` nisbiy yaxshilanish emas, balki accuracy orasidagi foiz punkt
farqidir. Dynamic model hali shu baseline bilan solishtirilmagan.

### Static model cheklovlari

- faqat bitta frame bilan ishlaydi;
- vaqt bo'yicha harakatni tushunmaydi;
- ikki qo'l sequence'ini to'liq ishlatmaydi;
- so'z va gapga birlashtirmaydi;
- kamera burchagi o'zgarganda umumlashtirish kafolatlanmagan.

---

## 2. Kamera pipeline'ida bajarilgan o'zgarishlar

O'zgartirilgan fayl:

- `pose/webcam_predict.py`

Qo'shilgan va tuzatilgan imkoniyatlar:

- eski model path o'rniga loyiha ichidagi model path ishlatiladi;
- model format va 42 feature soni tekshiriladi;
- confidence hisoblanadi;
- past confidence uchun `Noaniq` holati chiqariladi;
- oxirgi prediction'lar bo'yicha smoothing qo'shildi;
- RTMLib `lightweight` modeli ishlatildi;
- kamera resolution `640x480` qilindi;
- kamera buffer `1` ga tushirildi;
- pose inference har 2-kadrda bajariladigan qilindi;
- kamera ochilmasa aniq xato beriladi;
- feature extraction alohida funksiyaga ajratildi.

### Bu qism bo'yicha foiz

Kamera tezligi yoki latency bo'yicha oldin/keyin benchmark yozib olinmagani
uchun foizli tezlik yaxshilanishi **tasdiqlanmagan**. Shuning uchun bu qismga
sun'iy `50%` yoki `2x` raqam berilmadi.

Tasdiqlangan texnik o'zgarish:

```text
Pose inference: har bir frame -> har 2-frame
```

Bu nazariy pose inference yuklamasini taxminan 50% gacha kamaytirishi mumkin,
lekin real FPS va latency qurilma benchmarki bilan alohida o'lchanishi kerak.

---

## 3. Dependency va muhit

`requirements.txt` to'ldirildi:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `opencv-python`
- `onnxruntime`
- `rtmlib`
- `torch`

Ishlatilgan muhit:

```text
Python 3.13
/usr/local/bin/python3.13
```

Tekshirilgan importlar:

```text
cv2       OK
numpy     OK
sklearn   OK
torch     OK
rtmlib    OK
```

RTMLib lightweight modellarining yuklanishi ham muvaffaqiyatli tekshirildi.

---

## 4. Feature formatidagi o'zgarishlar

Yangi fayl:

- `features/temporal_features.py`

Har bir frame uchun:

- chap qo'l 21 landmark;
- o'ng qo'l 21 landmark;
- lokal `x/y` koordinatalar;
- confidence;
- wrist normalization;
- palm-scale normalization.

Natija:

```text
Raw frame feature: 126
Motion bilan feature: 378
```

378 feature quyidagidan tashkil topadi:

```text
126 original feature
+ 126 velocity
+ 126 acceleration
= 378
```

Bu dynamic modelga faqat qo'l shaklini emas, harakat yo'nalishi va tezligini
ham ko'rish imkonini beradi.

---

## 5. Dynamic sequence recorder

Yangi fayl:

- `pose/dynamic_recorder.py`

Ishga tushirish:

```bash
/usr/local/bin/python3.13 pose/dynamic_recorder.py \
  --label salom \
  --signer signer_01 \
  --view front
```

Boshqaruv:

- `Space` — recording boshlash;
- `Space` — recording tugatish;
- `Q` — chiqish.

Saqlanadigan fayllar:

```text
data/dynamic/sequences/*.npz
data/dynamic/metadata.csv
```

Metadata:

- sequence ID;
- label;
- signer ID;
- view;
- frame count.

Bu recorder oldingi 6000+ static datasetni o'zgartirmaydi va uni qayta
yig'ishni talab qilmaydi.

---

## 6. Dynamic Deep Learning modeli

Yangi fayl:

- `training/train_dynamic.py`

Model arxitekturasi:

```text
378 feature
    ↓
Linear projection
    ↓
Sinusoidal positional encoding
    ↓
3-layer Temporal Transformer
    ↓
Masked temporal pooling
    ↓
Classifier
```

Qo'shilgan texnik himoyalar:

- sequence padding mask;
- maksimal 512 frame limiti;
- gradient clipping;
- signer-based train/test split;
- train split'da barcha klasslar borligini tekshirish;
- model checkpoint ichida feature version;
- model checkpoint ichida label mapping;
- model checkpoint ichida input size;
- data bo'lmasa aniq error.

Model parametrlari test qilindi:

```text
Forward pass: OK
Input: [2, 12, 378]
Output: [2, 3]
```

Dynamic model accuracy:

```text
Hali o'lchanmagan
```

Sabab: `data/dynamic/metadata.csv` va dynamic sequence'lar hali to'liq
yig'ilmagan.

---

## 7. Dynamic webcam inference

Yangi fayl:

- `pose/webcam_dynamic.py`

Imkoniyatlar:

- rolling sequence window;
- ikki qo'l landmarklari;
- velocity va acceleration;
- lightweight RTMLib;
- pose inference frame skipping;
- confidence threshold;
- oxirgi prediction'lar smoothing'i;
- `Noaniq` natijani ko'rsatish;
- `Q` bilan xavfsiz chiqish.

Ishga tushirish:

```bash
/usr/local/bin/python3.13 pose/webcam_dynamic.py
```

Qo'shimcha parametrlar:

```bash
--window 32
--process-every 2
--threshold 0.65
```

Bu modul model o'qitilmasdan ishlamaydi. Avval dynamic data yig'ilib,
`dynamic_transformer.pt` yaratilishi kerak.

---

## 8. Static collector va split path tuzatishlari

O'zgartirilgan fayllar:

- `pose/live_collector.py`
- `pose/video_labeler.py`
- `training/split_dataset.py`

Natija:

- scriptlar qo'l feature'larini bir xil formatda olishga yaqinlashtirildi;
- `split_dataset.py` path'lari script joylashuviga bog'landi;
- loyiha root'idan ishlatish talab qilinmaydi;
- mavjud static `data/dataset.csv` ishlatiladi.

---

## 9. Dokumentatsiya va reja

Yaratilgan/yangilangan fayllar:

- `plan.md` — to'liq rivojlantirish rejasi;
- `README.md` — dynamic recorder va training buyruqlari;
- `complete.md` — ushbu bajarilgan ishlar hisoboti.

`plan.md` ichida:

- ikki qo'l pipeline;
- yuz va tana landmarklari;
- dynamic dataset;
- Transformer/TCN/BiLSTM variantlari;
- signer-based split;
- confidence;
- unknown gesture rejection;
- phrase va sentence decoder;
- MacBook benchmark;
- smart glasses port rejasi;

keltirilgan.

---

## 10. Test va verification

Muvaffaqiyatli tekshiruvlar:

```text
Python syntax errors: yo'q
Dynamic feature shape: 126
Motion feature shape: 378
Dynamic Transformer forward pass: OK
Dynamic CLI --help: OK
Dynamic recorder CLI --help: OK
Existing static model loading: OK
Existing static test accuracy: 93.29%
RTMLib lightweight initialization: OK
git diff --check: OK
```

Hali bajarilmagan verification:

- dynamic sequence dataset bo'yicha training;
- unseen signer accuracy;
- unseen view accuracy;
- real-time dynamic FPS;
- end-to-end Uzbek sentence accuracy;
- smart glasses hardware testi.

---

## 11. Foizlar bo'yicha halol hisobot

| Yo'nalish | O'lchangan natija | Izoh |
|---|---:|---|
| Static Random Forest accuracy | **93.29%** | 1311 test sample, 29 klass |
| Static MLP accuracy | **91.23%** | Alohida static baseline |
| RF va MLP farqi | **2.06 foiz punkt** | RF yuqori |
| Dynamic feature kengayishi | **3x** | 126 dan 378 gacha |
| Pose inference chastotasi | **2x kamroq** | Har frame emas, har 2-frame |
| Dynamic model accuracy | **Hali yo'q** | Dynamic data kerak |
| Camera FPS yaxshilanishi | **O'lchanmagan** | Benchmark kerak |
| Sentence translation accuracy | **Hali yo'q** | Sentence dataset kerak |
| Smart glasses tayyorligi | **0% production** | MacBook pipeline hali yakunlanmagan |

“2x kuchliroq” yoki “2x aniqroq” degan da'vo faqat bir xil test setda
benchmark o'tkazilgandan keyin qilinadi. Hozir bunday o'lchov yo'q.

---

## 12. Hozirgi keyingi majburiy qadamlar

### A. Dynamic data yig'ish

Avval 10–20 ta Uzbek so'z:

```text
salom
rahmat
men
sen
u
yaxshi
yomon
yordam
kerak
ha
yo'q
suv
ovqat
ism
nima
qayerda
tushundim
tushunmadim
```

Har bir so'z uchun:

- kamida 100 ta sequence;
- kamida 2–3 signer;
- front, left, right view;
- sekin va tez ijro;
- turli masofa va yorug'lik.

### B. Dynamic training

```bash
/usr/local/bin/python3.13 training/train_dynamic.py
```

### C. Dynamic camera

```bash
/usr/local/bin/python3.13 pose/webcam_dynamic.py
```

### D. Benchmark

Dynamic model baseline bilan quyidagilar bo'yicha solishtiriladi:

- accuracy;
- macro F1;
- unseen signer;
- unseen view;
- confidence calibration;
- FPS;
- latency;
- false positive rate.

---

## Yakuniy holat

Static baseline saqlangan va tekshirilgan. Dynamic Deep Learning pipeline'ining
asosiy kod skeleti yozilgan va runtime/syntax tekshiruvdan o'tgan. To'liq
dynamic accuracy yoki gap tarjimasi hali tayyor emas, chunki buning uchun
label'langan Uzbek dynamic sequence dataset zarur.

### Internetdan qo'shilgan public data

Internetdan Uzbek sohaga tegishli public manba topilib, loyiha ichiga alohida
saqlandi:

```text
data/external/mrkomiljon-uzbek-sign-language/data.pickle
```

Tasdiqlangan tarkib:

- 2800 ta sample;
- 28 ta alphabet source class;
- har bir sample 42 ta MediaPipe hand-landmark feature;
- label'lar source folder ID (`0`–`27`) ko'rinishida.

Manba:

<https://github.com/Mrkomiljon/uzbek-sign-language>

Bu static alphabet data bo'lgani uchun dynamic word/sentence modelning tayyor
o'rnini bosmaydi. Label ID'lari Uzbek harf nomlariga manba tomonidan ochiq
mapping qilinmagan; shu sababli u mavjud `data/dataset.csv` bilan avtomatik
aralashtirilmadi. Importer qo'shildi:

```bash
/usr/local/bin/python3.13 training/import_public_uzbek_dataset.py
```

Bu public datasetni `data/external/.../dataset.csv` ga source label'larini
saqlagan holda konvert qiladi.

### External dataset train/test tayyorligi

Yangi yuklangan `mrkomiljon` dataset original `dataset.csv` ga tegilmasdan
stratified va reproducible tarzda ajratildi:

- `train.csv`: **2,240 ta** sample, 28 klassning har biridan 80 tadan;
- `test.csv`: **560 ta** sample, 28 klassning har biridan 20 tadan;
- nisbat: **80% / 20%**;
- seed: **42**;
- exact duplicate feature row train va test orasiga bo'linmadi.

Split script:

- [split_external_uzbek_dataset.py](./training/split_external_uzbek_dataset.py)

Natijalar:

- [train.csv](./data/external/mrkomiljon-uzbek-sign-language/train.csv)
- [test.csv](./data/external/mrkomiljon-uzbek-sign-language/test.csv)

Numeric source label'lar (`0`-`27`) Uzbek harflariga tasdiqlangan mapping
bo'lmagani uchun asosiy named-label modelga avtomatik qo'shilmagan.

### Static model cleanup

Eski static model fayllari production candidate sifatida olib tashlandi:

- `models/gesture_classifier.pkl`;
- `models/gesture_extra_trees.pkl`;
- boshqa old static checkpointlar.

Datasetlar o'chirilmadi. Uzbek dynamic public checkpoint saqlandi:
`data/external/uzslr-isolated-dynamic/best_model.pth`. U 50 class, 32 frame va
MediaPipe Holistic 708-channel contractga ega. Uni oldingi 42-feature RTMLib
modeli bilan aralashtirib yuborish mumkin emas.

Hugging Face/Kaggle qidiruvida Uzbek alphabet uchun shu RTMLib contractiga
mos tayyor model topilmadi. Shuning uchun mavjud incompatible modelni
“5x/10x kuchliroq” deb noto'g'ri o'rnatish o'rniga datasetlar saqlandi va
checkpoint contracti aniq hujjatlashtirildi.

### Handex-style live tracking foundation

Handex sahifasining proprietary implementationi ochiq emas. Ochiq stack bilan
qayta yaratilayotgan live foundation quyidagilarni qo'shdi:

- RTMLib 133-keypoint hand indexlari markazlashtirildi;
- confidence-aware adaptive EMA smoothing;
- qisqa missing-frame hold va hand identity continuity;
- wrist-relative 42-feature extraction helperi;
- checkpoint landmark layout runtime validationi;
- live FPS overlay;
- alphabet-only scope; word/sentence decoding bu bosqichda yo'q.

Bu hali “Handex bilan teng” yoki 5x/10x yaxshiroq degani emas. Buni aytish
uchun bir xil live benchmarkda per-class accuracy, jitter, FPS va latency
o'lchanishi kerak.

External Random Forest experiment test CSV'da 100% ko'rsatgan bo'lsa-da,
live kamera landmark contracti mos kelmagani uchun production/live model
sifatida qabul qilinmadi va model fayli olib tashlandi. Public source data va
80/20 split saqlanib qoldi. Keyingi model RTMLib capture schema bilan bir xil
formatda train qilinishi kerak.

### Uzbek dynamic pretrained model

Qo'shimcha ravishda aynan Uzbek dynamic sign-language loyihasidan tayyor
checkpoint yuklandi:

```text
data/external/uzslr-isolated-dynamic/best_model.pth
```

Manba:

<https://github.com/akkomron/uzslr-isolated-dynamic>

Manba model:

- 50 ta Uzbek isolated sign;
- 32 frame sequence;
- MediaPipe Holistic;
- 708 preprocessed temporal channels;
- CNN + Transformer architecture;
- source README bo'yicha taxminan 92% validation va 87% test accuracy.

Checkpoint bu loyihaning static modeliga aralashtirilmagan, chunki input schema
butunlay boshqa. Uni yuklash uchun:

```python
from inference.uzslr_pretrained import load_pretrained_uzslr
model, labels = load_pretrained_uzslr()
```

Raw MediaPipe Holistic `(32, 1662)` sequence uchun loyiha ichidagi aniq
preprocessing adapteri ham qo'shildi:

- [uzslr_preprocess.py](./inference/uzslr_preprocess.py)
- [uzslr_isolated_dynamic_constants.py](./data/external/uzslr_isolated_dynamic_constants.py)
- `predict_raw_sequence(...)` orqali `(32, 708)` ga konvertatsiya va prediction.

Synthetic testda `(32, 1662) -> (32, 708) -> 50-class prediction` to'liq
muvaffaqiyatli o'tdi. Real kamera hali source MediaPipe Holistic landmark
oqimini talab qiladi; shu sababli random/static landmarklar bilan accuracy
da'vo qilinmaydi.

### Sign Language Dataset Hub catalog

`rudra496/SignLanguage-Dataset-Hub` repositorysi lokalga klon qilindi:

- [dataset hub](./data/external/sign-language-dataset-hub/)
- catalog: 66 verified rows, multiple languages and modalities;
- Uzbek entry: topilmadi;
- repository-included Bangla sensor demo yuklandi:
  - train: 3,528;
  - validation: 648;
  - test: 648.

Bu data Bangla sensor-glove formatida va Uzbek camera alphabet modeliga
qo'shilmagan. Qolgan datasetlar alohida source, credentials, registration,
research/commercial license va katta hajm talab qiladi; shuning uchun “all”
catalogni ko'r-ko'rona bulk-download qilish texnik yoki litsenziya jihatdan
to'g'ri emas. Har bir target dataset source/license tekshiruvidan keyin
alohida import qilinadi.

### `uzibytes/sign2text` tekshiruvi

Bu repository bizning maqsadga arxitektura jihatidan yaqin: MediaPipe Hands,
21 ta landmark, kamera oqimi va Random Forest. Ammo uning tayyor `model.p` va
`data.pickle` fayllari Uzbek model sifatida ishlatilmaydi:

- `data/0` dan `data/32` gacha 33 ta raqamli klass bor, Uzbek harflariga
  tasdiqlangan mapping yo'q;
- preprocessing MediaPipe Hands'ning normalized bounding-box koordinatalarini
  ishlatadi, bizning mavjud model esa RTMLib Wholebody 133-keypoint oqimidan
  olingan wrist-relative 42 feature kutadi;
- train/test random frame split bilan qilingan, bir xil video/kadrlar ikkala
  qismga tushishi mumkin, shuning uchun e'lon qilingan accuracy live yoki
  unseen-signer accuracy degani emas;
- `create_dataset.py` bir nechta qo'l bo'lsa feature uzunligini o'zgartirishi
  mumkin va `collect_imgs.py` faqat 100 ta kadr yig'adi;
- Flask/WebSocket qismi UI transportidir, aniqlikni oshiradigan model emas.

Foydali qismi — MediaPipe Hands'ning 21-landmark, bounding-box normalization,
kamera-native capture va WebSocket UI yondashuvi. Uni tayyor modelni ko'chirib
emas, balki mavjud Uzbek label mapping va camera calibration bilan
reimplement qilish kerak. Hozircha RTMLib'dan MediaPipe'ga aralashtirib o'tish
qilinmadi, chunki bu yangi feature contract va qayta dataset/training talab
qiladi.

Tekshiruvdan keyin alohida compatibility rejimi qo'shildi:

```bash
/usr/local/bin/python3.13 pose/webcam_predict.py --camera 1 --model sign2text
```

Bu rejim RTMLib 21 ta nuqtasini frame o'lchamiga bo'lib, source preprocessing
bo'yicha `(x - min_x, y - min_y)` qiladi. Asosiy Uzbek model default bo'lib
qoladi. Source model o'zining 2,849 sample'ida 100% qaytardi, lekin bu
in-sample natija; iPhone/live natija hali o'lchanmagan. Source klasslari
English A-Z va so'zlardan iborat, shuning uchun bu rejim Uzbek mapping
tasdiqlangan deb hisoblanmaydi.

Live inference'dagi noaniq harflar ham cheklab qo'yildi: top confidence kamida
55%, top-1/top-2 margin kamida 12 percentage point va candidate kamida 4
processed frame barqaror bo'lmasa, harf chiqarilmaydi. Qo'l yo'qolganda eski
prediction ekranda qolmaydi.

Qo'lda kamera data yig'masdan production model tayyorlandi: mavjud 6,547 ta
label'i aniq Uzbek sample'ning barchasi ExtraTrees bilan o'qitiladi va
`models/gesture_extra_trees_full.pkl` sifatida tanlanadi. Oldingi 80/20
held-out natija 93.75% bo'lib qoladi; full-data model uchun alohida test
accuracy da'vo qilinmaydi.

Bu source checkpoint uchun berilgan accuracy'lar mustaqil ravishda qayta
o'lchanmagan; ular source loyihaning o'z test natijalari.

Bu hisobotda faqat o'lchangan raqamlar foiz sifatida berildi. O'lchanmagan
qismlar ataylab “hali yo'q” yoki “o'lchanmagan” deb ko'rsatildi.
