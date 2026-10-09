import cv2
import csv
import ssl
import sys
from pathlib import Path
import numpy as np
from rtmlib import Wholebody

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.live_hand_tracking import choose_best_hand, extract_wrist_relative_features

# Mac uchun SSL muammosini chetlab o'tish
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    video_path = 'full_vidio.mp4'  # Sizning to'liq videongiz nomi
    csv_file = ROOT / "data" / "processed" / "legacy_video_capture.csv"
    
    # O'zbek daktil alifbosining ketma-ketligi
    letters = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'O_star', 'G_star', 'SH', 'CH', 'NG']
    current_idx = 0
    
    print("1. AI Model yuklanmoqda (Biroz kuting)...")
    openpose = Wholebody(to_openpose=False, backend='onnxruntime', device='cpu')
    
    # CSV faylni tayyorlash (Yangi toza baza ochamiz)
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}'])
        
    f = csv_file.open(mode='w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(header)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Xatolik: '{video_path}' topilmadi! Video kod turgan joyda ekanligiga ishonch hosil qiling.")
        return

    print("\n" + "="*50)
    print("🎥 INTERAKTIV DATA YIG'ISH BOSHQARUVI:")
    print(" [ S ] - Tugmasini BOSIB TURSANGIZ, kadrlar AI tomonidan saqlanadi.")
    print(" [ N ] - Tugmasi KEYINGI harfga o'tkazadi (A -> B -> D ...).")
    print(" [ BO'SHLIQ ] (Space) - Videoni pauza qilish / davom ettirish.")
    print(" [ Q ] - Dasturni to'xtatish.")
    print("="*50 + "\n")
    
    paused = False
    saved_count = 0
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video yakunlandi!")
                break
        
        display_frame = frame.copy()
        current_letter = letters[current_idx] if current_idx < len(letters) else "TUGADI"
        
        # Ekranga qulay interfeys chizish
        cv2.putText(display_frame, f"Harf: {current_letter}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(display_frame, f"Saqlandi: {saved_count} ta kadr", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "'S' ni bosib turing | 'N' keyingi harf", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('AI Video Labeler', display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Bo'shliq tugmasi
            paused = not paused
        elif key == ord('n'):  # Keyingi harfga o'tish
            current_idx += 1
            saved_count = 0
            if current_idx >= len(letters):
                print("Barcha harflar tugadi!")
                break
            print(f">>> Keyingi harfga o'tildi: {letters[current_idx]}")
        elif key == ord('s'):  # S tugmasi bosib turilganda
            if current_letter != "TUGADI" and not paused:
                # Kadrni analiz qilish
                keypoints, scores = openpose(frame)
                if len(keypoints) > 0:
                    kpts = keypoints[0]
                    kpts_scores = scores[0]
                    
                    hand_kpts, hand_scores, _ = choose_best_hand(kpts, kpts_scores)
                    if hand_kpts is None or float(np.min(hand_scores)) < 0.10:
                        continue 
                    
                    # Normallashtirish va saqlash
                    features = extract_wrist_relative_features(hand_kpts)
                    row = [current_letter]
                    row.extend(float(value) for value in features)
                    
                    writer.writerow(row)
                    saved_count += 1
                    print(f"[{current_letter}] saqlandi: {saved_count}")

    cap.release()
    cv2.destroyAllWindows()
    f.close()
    print("\n🎉 Tabriklaymiz! Barcha ma'lumotlar dataset.csv ga muvaffaqiyatli saqlandi.")

if __name__ == "__main__":
    main()