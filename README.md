# Uzbek Sign Language (USL) Real-Time Recognition System

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange?logo=scikit-learn&logoColor=white)
![RTMPose](https://img.shields.io/badge/Pose_Estimation-RTMLib-red)
![Machine Learning](https://img.shields.io/badge/AI-Classification-blueviolet)

O'zbek imo-ishora tili (USL) daktil alifbosini real vaqt rejimida (real-time inference) aniqlovchi va tasniflovchi sun'iy intellekt tizimi. Loyiha kompyuter ko'rishi (Computer Vision) va mashinali o'rganish (Machine Learning) algoritmlarini integratsiya qilgan holda, imo-ishoralarni matnga o'girish jarayonini to'liq avtomatlashtiradi.

## 📌 1. Muammo va Yechim (Problem Statement & Solution)

**Muammo:** O'zbekistonda eshitish va gapirish qobiliyati cheklangan insonlar uchun raqamli ochiq ma'lumotlar bazasi (open-source dataset) deyarli mavjud emas. Hozirda qo'llanilayotgan imo-ishora tillarida kirill va lotin yozuviga asoslangan daktil alifbolari o'rtasida o'tish jarayoni ketmoqda. Mavjud global modellar (masalan, ASL - American Sign Language) O'zbek daktil alifbosi morfologiyasi va barmoq joylashuv spetsifikatsiyalariga umuman mos kelmaydi.

**Yechim:** Yangi lotin yozuviga asoslangan (A-Z, O', G', Sh, Ch, Ng) 29 ta harfdan iborat daktil alifbosini to'liq taniy oladigan end-to-end Machine Learning pipeline ishlab chiqildi. Tizim yopiq muhitda (synthetic data) emas, balki jonli kadrlar asosida fine-tuning qilingan classifier yordamida yuqori aniqlikda (accuracy > 94%) ishlaydi. 

Bu shunchaki bazaviy deteksiya emas, balki har bir qo'l bo'g'imi (21 ta keypoint) orasidagi fazoviy bog'liqlikni (spatial relationship) hisoblaydigan va shovqinlardan (noise) tozalangan murakkab tasniflash tizimidir.

## ⚙️ 2. Arxitektura va DL Pipeline

Loyihaning arxitekturasi asosan ikki bosqichli model zanjiriga asoslangan: **Feature Extraction (Belgilarni ajratib olish)** va **Classification (Tasniflash)**.

```mermaid
graph TD
    A[Webcam Video Stream] --> B[RTMPose / Wholebody Model]
    B --> C{Keypoints Confidence > 0.2?}
    C -->|Yes| D[Hand Keypoint Extraction 21 pts]
    C -->|No| A
    D --> E[Spatial Normalization relative to Wrist]
    E --> F[Random Forest Classifier Inference]
    F --> G[Real-time Text Output]
````
Pose Estimation (RTMLib): Kadr yuzasidan inson tanasi va qo'l skeletlari onnxruntime backend orqali o'qiladi. Chap va o'ng qo'l tensorlari ajratilib, ishonchlilik ko'rsatkichi (confidence score > 0.2) eng yuqori bo'lgan qo'l tanlab olinadi.

Feature Engineering & Normalization: Model kadrning qayerida turishingizga qaram (overfit) bo'lib qolmasligi uchun, barcha 21 ta barmoq nuqtasi bilak (wrist) koordinatasiga nisbatan ayirilib, nolinchi o'qqa (0,0) normallashtirildi. Bu data distribution shift muammosini to'liq hal qildi.

Classification: Ekstraksiya qilingan 42 ta o'lchamli (X va Y koordinatalar) feature space Scikit-Learn'ning RandomForestClassifier (n_estimators=100) modeliga uzatiladi.

📊 3. Dataset Generatsiyasi va Data Pipeline
Loyiha uchun tayyor dataset yo'qligi sababli, ma'lumotlar bazasi noldan, ikki bosqichda yaratildi:

Stage 1 (Baseline Testing): Arxitektura va pipeline'ning uzluksiz ishlashini tekshirish uchun tasodifiy qiymatlardan (random float coordinates) iborat dummy dataset generatsiya qilindi va tizim integratsiyasi testdan o'tkazildi.

Stage 2 (Real Data Collection & Labeling): Modelni real muhitga moslashtirish uchun maxsus interaktiv skript (live_collector.py) yozildi. Kameraga qarab, har bir harf uchun jonli ravishda 21 ta barmoq bo'g'imining aniq koordinatalari yig'ildi. Natijada 6547 ta toza (labeled) kadr saqlab olindi. Oyna effekti (mirroring) sababli kelib chiqadigan xatoliklar kadrni cv2.flip qilish orqali bartaraf etildi va faqat aniq bitta o'q qoidalari datasetga yozildi.

📂 4. Loyiha Tuzilmasi (Repository Structure)
Plaintext
uzbek-sign-language-ai/
├── data/
│   ├── dataset.csv                 # 6547 qatorli tozalangan hand-keypoints bazasi
│   ├── sign_language_model.pkl     # O'qitilgan (Trained) Random Forest model
│   └── videos/                     # (Optional) Xom video ma'lumotlar
├── pose/
│   ├── dataset_builder.py          # Videolardan avtomatlashtirilgan data ajratuvchi
│   ├── live_collector.py           # Real-time interaktiv dataset yig'ish skripti (Custom Labeler)
│   ├── train_model.py              # ML modelni o'qitish va validatsiya qilish (Classifier Trainer)
│   └── webcam_predict.py           # Real-time Inference skripti (Live Detection)
├── requirements.txt                # Kerakli kutubxonalar ro'yxati
└── README.md                       # Loyiha hujjatlari
💻 5. Texnik Talablar (Hardware & Software Requirements)
Tizim inference jarayonida kechikishlar (latency) bo'lmasligi uchun optimizatsiya qilingan:

OS: Windows 10/11, macOS, yoki Linux.

RAM: Minimum 4GB (Model training va dataset RAM'da o'qilishi uchun).

CPU: Zamonaviy 4 yadroli protsessor (ONNX runtime backend CPU'da yaxshi ishlaydi).

GPU (Optional): Agar mavjud bo'lsa, RTMPose modelini CUDA/MPS orqali tezlashtirish mumkin, lekin CPU inference ham real-time (30+ FPS) ishlashga qodir.

Kamera: Minimum 720p HD veb-kamera (Yorug'likni yaxshi qabul qiluvchi linza tavsiya etiladi).

🚀 6. Tezkor Ishga Tushirish (Quick Start)
1. Repozitoriyni yuklab olish:

Bash
git clone [https://github.com/yourusername/uzbek-sign-language-ai.git](https://github.com/yourusername/uzbek-sign-language-ai.git)
cd uzbek-sign-language-ai
2. Virtual muhit va kutubxonalarni o'rnatish:

Bash
python3 -m venv .venv
source .venv/bin/activate  # Windows uchun: .venv\Scripts\activate
pip install -r requirements.txt
3. Modelni o'qitish (Model Training):
(Diqqat: Loyihada tayyor sign_language_model.pkl bo'lishiga qaramay, muhit o'zgarishlariga moslashish uchun uni locally qayta train qilish tavsiya etiladi)

Bash
python pose/train_model.py
Kutilayotgan natija: Accuracy metric 94% dan yuqori bo'lishi kerak.

4. Real-time Inference (Jonli efirda test qilish):

Bash
python pose/webcam_predict.py
⚠️ 7. Muammolarni Boshqarish (Troubleshooting)
Agar model ishlamasa yoki noto'g'ri (hallucination) tasnif qilsa:

Aniqlik keskin pasayishi (Model Misclassification): Bu odatda yorug'lik keskin o'zgarganda yoki kadrda ikkita qo'l aralashib ketganda yuzaga keladi (Data Distribution Shift). Yechim: Orqa fonni tozalang, xona yorug'ligini oshiring.

Harflarni umuman tanimasligi: Agar tizim butunlay tasodifiy (random) qiymat qaytarayotgan bo'lsa, model to'g'ri o'qitilmagan. Terminalda yana bir bor train_model.py ni ishga tushiring va dataset.csv bo'sh emasligiga ishonch hosil qiling.

Kamera ishga tushmasligi: webcam_predict.py dagi cv2.VideoCapture(0) indeksini 1 yoki 2 ga o'zgartirib ko'ring (ayniqsa tashqi kamera ishlatayotgan bo'lsangiz).

🔮 8. Kelajakdagi Rejalar (Roadmap)
Time-series model (LSTM yoki Transformer) integratsiyasi orqali statik harflarni emas, balki dinamik (harakatdagi) so'zlarni ham classification qilish.

Yig'ilgan dataset hajmini oshirish va Clustering algoritmlari orqali ma'lumotlar bazasidagi outlier (anomal) kadrlarni tozalash.

Modelni TensorFlow Lite formatiga eksport qilib, mobil ilova (Android/iOS) ga deploy qilish.
