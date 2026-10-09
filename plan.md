# Uzbek Sign Language AI — To‘liq rivojlantirish rejasi

## 1. Yakuniy maqsad

MacBook kamerasi orqali insonning o‘zbek imo-ishora tilidagi harakatlarini real vaqt rejimida:

```text
Qo‘l + yuz + tana harakati
        ↓
3D-ga yaqin landmark sequence
        ↓
Dynamic Deep Learning model
        ↓
Imo-ishora / token
        ↓
So‘z
        ↓
O‘zbekcha gap
        ↓
Ekrandagi matn va keyinchalik smart glasses overlay
```

Tizim:

- faqat bitta static frame emas, vaqt bo‘yicha harakatni tushunadi;
- chap va o‘ng qo‘lni birga kuzatadi;
- old, chap yon va o‘ng yon ko‘rinishlarga chidamli bo‘ladi;
- qo‘l shakli bilan birga harakat yo‘nalishi va tezligini ham ishlatadi;
- alohida gesture’larni so‘z va gapga birlashtiradi;
- ishonchsiz natijani matnga qo‘shmaydi;
- avval MacBook’da ishlaydi;
- keyin smart glasses hardware’iga ko‘chiriladi.

> Muhim cheklov: bitta oddiy kamera haqiqiy 3D va odamning butun orqa tomonini mukammal ko‘ra olmaydi. Monocular model `x`, `y`, taxminiy `z` va confidence beradi. Haqiqiy depth yoki occlusion uchun keyinchalik stereo/depth kamera yoki bir nechta ko‘rinish kerak bo‘ladi.

---

## 2. Hozirgi loyiha holati

### Mavjud qismlar

- `data/dataset.csv` — taxminan 6000+ static landmark namunasi;
- 29 ta static klass: harflar va Uzbek alifbosidagi maxsus belgilar;
- `training/train.py` — Random Forest baseline;
- `training/train_mlp.py` — PyTorch MLP baseline;
- `training/split_dataset.py` — train/test split;
- `training/visualize_results.py` — Random Forest grafiklari;
- `training/visualize_mlp.py` — MLP grafiklari;
- `pose/webcam_predict.py` — kamera orqali static prediction;
- RTMLib/ONNX pose detector;
- model checkpoint’lari `models/` ichida.

### Hozirgi model nimani qila oladi?

Hozirgi Random Forest/MLP:

- bitta frame’dan feature oladi;
- asosan bitta qo‘lni ishlatadi;
- static hand pose klassifikatsiya qiladi;
- dinamik gesture’ni vaqt bo‘yicha tushunmaydi;
- so‘z yoki gapga dekodlamaydi;
- smart glasses uchun tayyor production pipeline emas.

Mavjud `dataset.csv` o‘chirilmaydi. U:

1. static baseline sifatida;
2. hand geometry pretraining uchun;
3. feature extractor tekshiruvi uchun;
4. auxiliary static classification loss uchun;
5. keyingi dynamic modelni solishtirish uchun ishlatiladi.

---

## 3. Asosiy texnik qarorlar

### 3.1. Eski modelni tashlamaslik

Mavjud Random Forest va MLP o‘chirilmaydi. Ular baseline bo‘lib qoladi. Yangi model eski modeldan yaxshi yoki kamida amaliy jihatdan foydaliroq ekanini metrikalar bilan ko‘rsatishi kerak.

### 3.2. Dynamic Deep Learning

Yangi asosiy model:

```text
Landmark projection
    ↓
Temporal Transformer yoki TCN
    ↓
BiLSTM/Temporal head
    ↓
Gesture/word decoder
    ↓
CTC yoki sequence decoder
    ↓
O‘zbekcha text layer
```

Boshlang‘ich prototip uchun:

- PyTorch;
- temporal Transformer Encoder;
- masking/padding;
- CTC yoki isolated-word softmax;
- keyinchalik Transformer decoder.

LSTM/GRU sodda baseline sifatida yoziladi, lekin uzoq sequence uchun Transformer/TCN ham sinovdan o‘tkaziladi.

### 3.3. Feature format

Har bir frame uchun:

- chap qo‘l: 21 landmark;
- o‘ng qo‘l: 21 landmark;
- har bir landmark uchun `x`, `y`, taxminiy `z`, confidence;
- tanadan tanlangan pose landmarklari;
- yuzdan og‘iz/qosh kabi muhim landmarklar;
- missing landmark mask;
- frame timestamp;
- velocity;
- acceleration.

Feature’lar:

1. bilakka nisbatan local coordinate;
2. kaft o‘lchamiga nisbatan scale normalization;
3. tana markaziga nisbatan global position;
4. frame-to-frame velocity;
5. acceleration;
6. confidence va visibility;
7. aniqlanmagan nuqtalar uchun mask.

RTMLib indekslari bitta umumiy helper orqali boshqariladi. Qo‘l indekslari model formatiga mos ravishda markazlashtiriladi; collector va inference boshqa-boshqa slice ishlatmasligi kerak.

---

## 4. Dataset strategiyasi

### 4.1. Static dataset

`data/dataset.csv` saqlanadi va qayta yig‘ilmaydi.

Uning vazifasi:

- 29 static klassni baseline qilish;
- keypoint extractor to‘g‘ri ishlashini tekshirish;
- hand shape pretraining;
- static auxiliary task;
- regression test.

### 4.2. Dynamic dataset formati

Yangi dynamic ma’lumotlar video yoki to‘g‘ridan-to‘g‘ri landmark sequence sifatida saqlanadi:

```text
data/dynamic/
├── sequences/
│   ├── seq_000001.npz
│   └── seq_000002.npz
├── metadata.csv
├── labels.json
└── splits/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

Har bir sequence metadata:

```text
sequence_id
label
signer_id
view
session_id
fps
frame_count
feature_version
```

Video saqlash foydali, lekin trening uchun avval landmark sequence saqlash disk va computation xarajatini kamaytiradi. Muhim sample’lar uchun original video ham saqlanadi.

### 4.3. Dataset bosqichlari

#### Bosqich A — isolated dynamic words

Avval 10–20 ta so‘z:

- salom;
- rahmat;
- men;
- sen;
- u;
- yaxshi;
- yomon;
- yordam;
- kerak;
- ha;
- yo‘q;
- suv;
- ovqat;
- ism;
- nima;
- qayerda;
- tushundim;
- tushunmadim.

Har bir so‘z uchun:

- kamida 300–500 sequence;
- kamida 5–10 signer;
- front, chap yon, o‘ng yon ko‘rinish;
- turli masofa;
- turli yorug‘lik;
- sekin, normal va tez bajarish;
- boshlanish va tugashdagi transition;
- qo‘lning kadrdan qisman chiqib qolishi.

#### Bosqich B — phrase

Misollar:

- “Men yaxshiman.”
- “Menga yordam kerak.”
- “Sizni tushunmadim.”
- “Ismingiz nima?”
- “Qayerga ketyapsiz?”

#### Bosqich C — erkin gaplar

Model oldindan berilmagan gesture ketma-ketliklarini tokenlarga ajratib, gapga birlashtiradi.

### 4.4. Data augmentation

Landmark darajasida:

- kichik rotation;
- translation;
- scale;
- noise;
- time stretching;
- frame drop;
- speed change;
- random occlusion/missing points;
- left/right view variation;
- confidence pasayishi;
- sequence crop/pad.

Data augmentation haqiqiy geometriyani buzmasligi kerak. Harflar yoki gesture ma’nosini o‘zgartiradigan kuchli transformation ishlatilmaydi.

### 4.5. Train/validation/test split

Frame bo‘yicha random split qilinmaydi. Aks holda bir odamning juda o‘xshash kadrlari train va testga tushib, natija sun’iy yuqori chiqadi.

To‘g‘ri split:

```text
Train: signer 1–7
Validation: signer 8
Test: signer 9–10
```

Natijalar:

- isolated word accuracy;
- macro F1;
- har bir so‘z uchun precision/recall;
- unseen signer accuracy;
- unseen view accuracy;
- latency va FPS;
- unknown gesture rejection;
- confusion matrix.

---

## 5. Landmark extraction pipeline

### 5.1. Boshlang‘ich backend

Hozirgi RTMLib MacBook prototipida qoladi. Real-time tezlik uchun:

- lightweight model;
- kamera resolution 640×480 yoki qurilma imkoniga mos;
- frame buffer 1;
- pose inference har 2–3 frame’da;
- ekran har frame’da yangilanadi;
- oldingi landmark qisqa vaqt qayta ishlatiladi.

### 5.2. Alternativ backend

MediaPipe Hand/Pose/Face Landmarker ham sinovdan o‘tkaziladi. Qaysi biri:

- yuqori FPS;
- kam latency;
- kamroq jitter;
- qo‘l confidence’i yuqori;
- ikki qo‘lni barqaror topishi;

bo‘yicha yaxshi bo‘lsa, MacBook real-time backend sifatida tanlanadi.

### 5.3. Har bir frame pipeline

```text
Camera frame
    ↓
Resize/flip
    ↓
Person detection
    ↓
Left/right hand detection
    ↓
Pose/face landmarks
    ↓
Quality gate
    ↓
Normalization
    ↓
Velocity/acceleration
    ↓
Sequence buffer
```

Quality gate:

- qo‘l confidence’i threshold’dan yuqori;
- minimal landmark confidence;
- qo‘l bounding box juda kichik yoki juda katta emas;
- landmark koordinatalari finite;
- frame’da kamida bitta valid hand;
- kerak bo‘lsa ikki qo‘l mavjudligi.

Past sifatli frame modelga “haqiqat” sifatida berilmaydi; mask yoki reject qilinadi.

---

## 6. Dynamic model bosqichlari

### 6.1. Model 0 — baseline

Hozirgi static Random Forest/MLP natijasi saqlanadi.

### 6.2. Model 1 — isolated word classifier

Input:

```text
[batch, time, feature_dim]
```

Architecture:

```text
Linear projection
    ↓
Positional encoding
    ↓
Transformer Encoder
    ↓
Temporal pooling
    ↓
Linear classifier
```

Loss:

- cross entropy;
- class weights;
- optional auxiliary static loss.

### 6.3. Model 2 — CTC sequence decoder

Model frame-level hidden representationdan token chiqaradi:

```text
frame sequence → token sequence
```

CTC:

- gesture boundary oldindan aniq belgilanmasa ham ishlaydi;
- real-time prefix decoding uchun qulay;
- repeated token va blank tokenni boshqaradi.

### 6.4. Model 3 — gap decoder

```text
gesture tokens
    ↓
word segmentation
    ↓
Uzbek vocabulary/grammar layer
    ↓
gap va punctuation
```

Til qatlami modelning ko‘rmagan ma’nosini o‘zboshimchalik bilan taxmin qilmasligi kerak. Past confidence tokenlar “noaniq” sifatida saqlanadi.

---

## 7. Gapga birlashtirish

Real-time text layer quyidagilarni boshqaradi:

- current gesture;
- gesture start/end;
- stable prediction;
- duplicate suppression;
- word boundary;
- pause detection;
- confidence threshold;
- user confirm;
- clear/backspace;
- punctuation.

Masalan:

```text
MEN + YAXSHI
        ↓
Men yaxshiman.
```

Bu grammatik conversion keyingi bosqichda qoida-based va model-based variantlarda solishtiriladi.

---

## 8. MacBook real-time interfeys

Ekranda ko‘rsatiladi:

- kamera;
- skeleton;
- chap/o‘ng qo‘l confidence;
- current gesture;
- top-3 prediction;
- FPS;
- latency;
- current word;
- current sentence;
- “Noaniq” holati;
- recorder/train debug mode.

Debug rejimida:

- feature dimension;
- keypoint min/mean score;
- selected hand;
- sequence length;
- model version;
- dropped frame count;

ko‘rsatiladi.

---

## 9. Test va sifat talablari

### Functional tests

- ikki qo‘l aniqlanadi;
- bir qo‘l yo‘q bo‘lsa pipeline yiqilmaydi;
- past confidence’da noto‘g‘ri text chiqmaydi;
- camera disconnect aniq xato beradi;
- checkpoint feature version tekshiriladi;
- sequence padding/masking tekshiriladi;
- empty sequence reject qilinadi.

### Model quality gates

Bosqichdan keyingi bosqichga o‘tish uchun:

1. validation natijasi testga yaqin bo‘lishi;
2. unseen signer testida katta pasayish bo‘lmasligi;
3. eng yomon klasslar alohida tekshirilishi;
4. `unknown` gesture reject ishlashi;
5. real kamera testida minimal FPS va latency bajarilishi.

Aniq thresholdlar dataset hajmiga qarab belgilanadi. Accuracy’ni faqat random frame split bilan baholash taqiqlanadi.

---

## 10. Performance optimizatsiyasi

### Mac prototip

- lightweight pose model;
- ONNX Runtime;
- CPU/MPS benchmark;
- frame skipping;
- inference thread;
- camera buffer 1;
- kichik input resolution;
- cached model;
- batch size 1;
- sequence window’ni cheklash.

### Real-time target

Avval o‘lchanadigan target:

- kamida 15 FPS inference;
- UI 25–30 FPS;
- gesture latency 200–400 ms oralig‘ida;
- memory leak bo‘lmasligi;
- kamera buffer kechikishi minimal bo‘lishi.

Natija qurilma va modelga qarab qayta belgilanadi.

---

## 11. Smart glasses’ga ko‘chirish

MacBook versiyasi quyidagilarni bajarmaguncha hardware port qilinmaydi:

- unseen signer’da ishlashi;
- bir nechta view’da ishlashi;
- stable sequence decoding;
- real-time latency o‘lchanishi;
- text layer alohida modul bo‘lishi;
- model export formati aniqlanishi;
- input/output contract hujjatlashtirilishi.

Keyin:

1. glasses kamera stream’i aniqlanadi;
2. inference MacBook/serverda qoladimi yoki edge device’ga ko‘chadimi — tanlanadi;
3. ONNX/TorchScript/CoreML export qilinadi;
4. matn glasses display’iga yuboriladi;
5. battery, heat, latency va network uzilishi sinovdan o‘tkaziladi;
6. offline fallback qo‘shiladi.

Ko‘zoynak kodi model pipeline’dan alohida adapter bo‘lishi kerak:

```text
CameraAdapter
LandmarkAdapter
SequenceModel
TextDecoder
DisplayAdapter
```

---

## 12. Fayl va modul tuzilmasi

Rejalashtirilgan tuzilma:

```text
data/
├── dataset.csv
└── dynamic/
    ├── sequences/
    ├── metadata.csv
    └── splits/

pose/
├── extractor.py
├── landmark_pipeline.py
├── dynamic_recorder.py
└── webcam_predict.py

features/
├── normalize.py
├── temporal_features.py
└── schema.py

models/
├── gesture_classifier.pkl
├── gesture_mlp.pt
└── dynamic_transformer.pt

training/
├── split_dataset.py
├── train.py
├── train_mlp.py
├── train_dynamic.py
├── evaluate_dynamic.py
└── visualize_results.py

inference/
├── predictor.py
├── sequence_decoder.py
└── text_decoder.py

api/
└── main.py
```

Har bir model checkpoint quyidagilarni saqlaydi:

- model weights;
- feature schema version;
- label/token list;
- normalization parameters;
- input dimension;
- class/token mapping;
- training seed;
- dataset version;
- model version.

---

## 13. Amalga oshirish ketma-ketligi

### Phase 0 — mavjud kodni barqarorlashtirish

- `dataset.csv` va mavjud modelni saqlash;
- feature schema’ni markazlashtirish;
- RTMLib index mapping’ni bitta joyga chiqarish;
- static camera baseline’ni to‘g‘rilash;
- FPS/latency log qo‘shish;
- dependency va interpreter hujjatlash.

### Phase 1 — dynamic recorder

- ikki qo‘l landmark recorder;
- pose va face tanlash;
- sequence start/stop;
- label/signer/view metadata;
- NPZ/JSON schema;
- recording quality preview.

### Phase 2 — 10–20 so‘z

- dataset yig‘ish;
- signer-based split;
- augmentation;
- isolated word Transformer/GRU;
- confusion matrix va real camera demo.

### Phase 3 — production-quality word model

- ko‘proq signer;
- ko‘proq view;
- unknown rejection;
- confidence calibration;
- quantization/ONNX benchmark;
- real-time stable decoder.

### Phase 4 — phrase model

- phrase sequence data;
- word boundary;
- CTC/decoder;
- Uzbek text normalization;
- pause va punctuation.

### Phase 5 — erkin gap

- token-level decoding;
- language layer;
- grammar correction;
- error recovery;
- user confirmation.

### Phase 6 — glasses port

- camera/display adapter;
- edge/server deployment;
- network fallback;
- battery/latency profiling;
- hardware acceptance tests.

---

## 14. Nimalarni qilmaslik kerak

- mavjud `dataset.csv` ni o‘chirmaslik;
- static data bilan dynamic gap modelini “tayyor” deb hisoblamaslik;
- random frame split bilan yuqori accuracy’ga aldanmaslik;
- bir qo‘l bilan ikki qo‘lli gesture’ni to‘liq tanishga urinish;
- past confidence predictionni gapga qo‘shmaslik;
- kamera resolution muammosini model sifati deb noto‘g‘ri talqin qilmaslik;
- hardware portni MacBook prototipi tekshirilmasdan boshlamaslik;
- checkpoint schema’siz model saqlamaslik;
- faqat bitta odamda test qilib production qaror chiqarmaslik.

---

## 15. Yakuniy natija mezoni

Loyiha “tayyor” deyilishi uchun:

- MacBook kamerasida ikki qo‘l va relevant face/pose landmarklari olinadi;
- model dynamic sequence’ni real-time tushunadi;
- kamida bir nechta unseen signer’da ishlaydi;
- front/chap/o‘ng ko‘rinishlarda testdan o‘tadi;
- gesture’lar so‘zga, so‘zlar gapga birlashadi;
- past confidence’da noto‘g‘ri gap chiqarmaydi;
- FPS va latency o‘lchangan bo‘ladi;
- model checkpoint reproducible bo‘ladi;
- mavjud static baseline bilan solishtirish mavjud bo‘ladi;
- keyinchalik glasses adapter’iga ulash uchun API contract tayyor bo‘ladi.

## Yakuniy qaror

Mavjud 6000+ static data saqlanadi va foydali baseline sifatida ishlatiladi. Asosiy rivojlanish yo‘li esa yangi dynamic landmark sequence dataset, ikki qo‘l + yuz + tana feature’lari va temporal Deep Learning modelidir. Avval MacBook’da ishonchli va o‘lchanadigan tizim quriladi; smart glasses faqat shu pipeline barqaror bo‘lgandan keyin ulanadi.
