import cv2
import ssl
from rtmlib import Wholebody, draw_skeleton

# Mac'dagi SSL sertifikati xatoligini chetlab o'tish
ssl._create_default_https_context = ssl._create_unverified_context

print("AI Model yuklanmoqda (bu birinchi marta biroz vaqt olishi mumkin)...")

# RTMW modelini chaqirish (onnxruntime orqali CPU/Mac'da ishlaydi)
openpose = Wholebody(to_openpose=False, backend='onnxruntime', device='cpu')
print("Model muvaffaqiyatli yuklandi!")

# Test rasmini o'qish
img_path = 'data/raw/test.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Xatolik: {img_path} rasmi topilmadi! Iltimos rasm borligini tekshiring.")
else:
    print("Rasm tahlil qilinmoqda...")
    # Model orqali bashorat qilish (qo'l, yuz va tana nuqtalarini topish)
    keypoints, scores = openpose(img)

    # Natijani rasmga chizish
    img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.3)
    
    # Rasmni ekranda ko'rsatish
    print("Natija ekranga chiqarildi. Oynani yopish uchun klaviaturada biron tugmani bosing.")
    cv2.imshow('Natija - Sign Language AI', img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()