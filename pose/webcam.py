import cv2
import ssl
from rtmlib import Wholebody, draw_skeleton

# Mac uchun xavfsizlik cheklovini aylanib o'tish (xuddi oldingidek)
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    print("AI Model yuklanmoqda...")
    # Model faqat bir marta, tsikl boshlanishidan oldin yuklanadi
    openpose = Wholebody(to_openpose=False, backend='onnxruntime', device='cpu')
    print("Kamera ishga tushirilmoqda...")

    # Mac dagi veb-kamerani ishga tushirish (0 - standart kamera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Xatolik: Kamera topilmadi yoki unga ruxsat yo'q!")
        return

    print("Kamera ishlamoqda. Dasturni to'xtatish uchun ekranda 'q' tugmasini bosing.")

    while True:
        # Kameradan bitta kadr (rasm) olish
        success, frame = cap.read()
        if not success:
            print("Kaderni o'qib bo'lmadi. Kamera uzilgan bo'lishi mumkin.")
            break

        # Kaderni oynaga moslashtirib teskari (oyna effekti) qilib qo'yish (ixtiyoriy)
        frame = cv2.flip(frame, 1)

        # AI orqali kadrni tahlil qilish
        keypoints, scores = openpose(frame)

        # Topilgan skelet/barmoq nuqtalarini shu kadrning o'ziga chizish
        frame_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)

        # Natijani jonli ravishda ekranga chiqarish
        cv2.imshow('Sign Language AI - Jonli', frame_show)

        # 'q' tugmasi bosilganda oynani yopish
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kamera va oynalarni toza yopish
    cap.release()
    cv2.destroyAllWindows()
    print("Dastur to'xtatildi.")

if __name__ == "__main__":
    main()