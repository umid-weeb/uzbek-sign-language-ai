import cv2
import csv
import ssl
import numpy as np
from rtmlib import Wholebody, draw_skeleton

ssl._create_default_https_context = ssl._create_unverified_context

def main():
    csv_file = 'data/dataset.csv'
    letters = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'O_star', 'G_star', 'SH', 'CH', 'NG']
    
    print("AI Model yuklanmoqda...")
    openpose = Wholebody(to_openpose=False, backend='onnxruntime', device='cpu')
    
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}'])
        
    f = open(csv_file, mode='w', newline='')
    writer = csv.writer(f)
    writer.writerow(header)
    
    cap = cv2.VideoCapture(0)
    current_idx = 0
    
    print("\n" + "="*50)
    print("🎥 JONLI DATA YIG'ISH:")
    print(" [ S ] - Bosib tursangiz, o'ng qo'lingiz kadrlarini yig'adi.")
    print(" [ N ] - Keyingi harfga o'tish.")
    print(" [ Q ] - Tugatish / Chiqish.")
    print("="*50 + "\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Oyna effekti
        display_frame = frame.copy()
        current_letter = letters[current_idx] if current_idx < len(letters) else "TUGADI"
        
        cv2.putText(display_frame, f"Hozirgi Harf: {current_letter}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(display_frame, "'S' ni bosib turing | 'N' keyingisi", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        display_frame = draw_skeleton(display_frame, *openpose(frame), kpt_thr=0.2)
        cv2.imshow('Live Data Collector', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_idx += 1
            if current_idx >= len(letters):
                print("Barcha harflar yig'ildi!")
                break
            print(f">>> Keyingi harf: {letters[current_idx]}")
        elif key == ord('s'):
            if current_letter != "TUGADI":
                keypoints, scores = openpose(frame)
                if len(keypoints) > 0:
                    kpts = keypoints[0]
                    kpts_scores = scores[0]
                    # Faqat o'ng qo'l nuqtalarini olamiz
                    hand_kpts = kpts[112:133]
                    
                    wrist_x, wrist_y = hand_kpts[0]
                    row = [current_letter]
                    for (x, y) in hand_kpts:
                        row.extend([round(float(x - wrist_x), 4), round(float(y - wrist_y), 4)])
                    
                    writer.writerow(row)
                    print(f"[{current_letter}] kaft koordinatasi saqlandi!")

    cap.release()
    cv2.destroyAllWindows()
    f.close()
    print("🎉 Yangi toza baza tayyorlandi!")

if __name__ == "__main__":
    main()