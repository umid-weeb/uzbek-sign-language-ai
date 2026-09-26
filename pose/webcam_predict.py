import cv2
import ssl
import pickle
import numpy as np
from rtmlib import Wholebody, draw_skeleton

# Mac uchun SSL muammosini chetlab o'tish
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    print("1. Ko'z (Pose Model) yuklanmoqda...")
    openpose = Wholebody(to_openpose=False, backend='onnxruntime', device='cpu')
    
    print("2. Miya (AI Classifier) yuklanmoqda...")
    with open('data/sign_language_model.pkl', 'rb') as f:
        classifier = pickle.load(f)
        
    cap = cv2.VideoCapture(0)
    print("Kamera ishga tushdi! Dasturni to'xtatish uchun 'q' ni bosing.")

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Videodagidek bir xil chiqishi uchun kamerani gorizontal oyna qilamiz
        frame = cv2.flip(frame, 1)
        
        # 1-qadam: Skeletni topish
        keypoints, scores = openpose(frame)
        frame_show = frame.copy()
        
        if len(keypoints) > 0:
            kpts = keypoints[0]
            kpts_scores = scores[0]
            
            # Qaysi qo'l aniqroq ko'rinayotganini tekshiramiz
            left_score = np.mean(kpts_scores[91:112])
            right_score = np.mean(kpts_scores[112:133])
            
            hand_kpts = None
            if right_score > left_score and right_score > 0.2:
                hand_kpts = kpts[112:133]
            elif left_score > right_score and left_score > 0.2:
                hand_kpts = kpts[91:112]
                
            if hand_kpts is not None:
                # Ekranga skeletni chizish
                frame_show = draw_skeleton(frame_show, keypoints, scores, kpt_thr=0.2)
                
                # 2-qadam: Koordinatalarni bilakka (0-chi nuqtaga) nisbatan tozalash
                wrist_x, wrist_y = hand_kpts[0]
                features = []
                for (x, y) in hand_kpts:
                    features.extend([round(float(x - wrist_x), 4), round(float(y - wrist_y), 4)])
                
                # 3-qadam: Bashorat qilish
                prediction = classifier.predict([features])[0]
                
                # Natijani yashil rangda ekranga katta qilib yozish
                cv2.putText(frame_show, f"Harf: {prediction}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        cv2.imshow('Sign Language AI - Live', frame_show)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()