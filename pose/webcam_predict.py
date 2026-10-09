import cv2
import ssl
import pickle
import sys
import time
import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np
from rtmlib import Wholebody, draw_skeleton

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.live_hand_tracking import (
    HandLandmarkSmoother,
    extract_bounding_box_features,
    extract_wrist_relative_features,
    validate_rtmlib_output,
)
from pose.camera_utils import open_camera, list_cameras, resolve_camera_index

# Mac uchun SSL muammosini chetlab o'tish
ssl._create_default_https_context = ssl._create_unverified_context

# ExtraTrees uses the same 42-feature RTMLib contract and is selected after
# outperforming the previous Random Forest on the fixed test split.
CALIBRATED_MODEL_PATH = ROOT / "models" / "gesture_extra_trees_camera.pkl"
FULL_MODEL_PATH = ROOT / "models" / "gesture_extra_trees_full.pkl"
BASE_MODEL_PATH = ROOT / "models" / "gesture_extra_trees.pkl"
MODEL_PATH = (
    CALIBRATED_MODEL_PATH
    if CALIBRATED_MODEL_PATH.exists()
    else FULL_MODEL_PATH
    if FULL_MODEL_PATH.exists()
    else BASE_MODEL_PATH
)
# Do not display low-confidence guesses as letters. A prediction must pass both
# confidence and margin gates and remain stable across several processed frames.
CONFIDENCE_THRESHOLD = 0.55
MIN_CONFIDENCE_MARGIN = 0.12
SMOOTHING_WINDOW = 9
MIN_STABLE_FRAMES = 4
INFERENCE_SIZE = (640, 480)
PROCESS_EVERY_N_FRAMES = 2
# RTMW scores for hand landmarks are commonly below 0.45 on Mac CPU
# lightweight inference. The previous values rejected every live frame.
MIN_HAND_SCORE = 0.10
MIN_KEYPOINT_SCORE = 0.05
SIGN2TEXT_LABELS = {
    str(index): label
    for index, label in enumerate(
        [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z", "Hello", "Done",
            "Thank You", "I Love You", "Sorry", "Please", "You are welcome",
        ]
    )
}


def load_classifier(path):
    if not path.exists():
        raise FileNotFoundError(f"Model topilmadi: {path}")
    with path.open("rb") as file:
        checkpoint = pickle.load(file)

    if isinstance(checkpoint, dict):
        classifier = checkpoint.get("model")
        feature_columns = checkpoint.get("feature_columns")
    else:
        classifier = checkpoint
        feature_columns = None

    if classifier is None or not hasattr(classifier, "predict"):
        raise ValueError(f"Model formati noto'g'ri: {path}")

    if getattr(classifier, "n_features_in_", 42) != 42:
        raise ValueError(
            "Kamera featurelari 42 ta bo'lishi kerak, "
            f"model esa {classifier.n_features_in_} ta kutmoqda."
        )

    if feature_columns is not None and len(feature_columns) != 42:
        raise ValueError("Modeldagi feature_columns 42 ta emas.")

    layout = checkpoint.get("keypoint_layout") if isinstance(checkpoint, dict) else None
    feature_mode = checkpoint.get("feature_mode") if isinstance(checkpoint, dict) else None
    if path.name == "model.p":
        if layout is not None:
            raise ValueError("sign2text model metadata'si noto'g'ri.")
    else:
        if layout != "rtmlib_mmpose_wholebody_v1":
            raise ValueError(f"Model landmark layout mos emas: {layout}")
        if feature_mode not in (None, "wrist_relative"):
            raise ValueError(f"Model feature contract mos emas: {feature_mode}")
    return classifier


def display_label(classifier, index: int, model_name: str) -> str:
    label = str(classifier.classes_[index])
    if model_name == "sign2text":
        return SIGN2TEXT_LABELS.get(label, label)
    return label


def main():
    parser = argparse.ArgumentParser(description="Run Uzbek alphabet recognition from Mac/iPhone camera.")
    parser.add_argument(
        "--camera",
        type=float,
        default=0,
        help="Camera index: 0=MacBook, 1=iPhone; 0.5=iPhone alias.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="Show available camera indexes and exit.")
    parser.add_argument(
        "--model",
        choices=("uzbek", "sign2text"),
        default="uzbek",
        help="uzbek: local Uzbek model; sign2text: imported MediaPipe-compatible model.",
    )
    args = parser.parse_args()
    if args.list_cameras:
        print("Mavjud camera indexlar:", list_cameras())
        return
    print("1. Ko'z (Pose Model) yuklanmoqda...")
    openpose = Wholebody(
        mode="lightweight",
        to_openpose=False,
        backend="onnxruntime",
        device="cpu",
    )
    print("2. Miya (AI Classifier) yuklanmoqda...")
    if args.model == "sign2text":
        model_path = ROOT / "data" / "external" / "sign2text" / "model.p"
        feature_mode = "bounding_box"
    else:
        model_path = MODEL_PATH
        feature_mode = "wrist_relative"
    classifier = load_classifier(model_path)
    print(f"Model: {model_path}")
        
    camera_index = resolve_camera_index(args.camera)
    try:
        cap = open_camera(camera_index, *INFERENCE_SIZE)
    except RuntimeError as error:
        if args.camera == 0.5:
            raise RuntimeError(
                "iPhone Continuity Camera index 1 ochilmadi. "
                "iPhone qulfdan chiqarilganini, Continuity Camera yoqilganini "
                "va boshqa dastur kamera ishlatmayotganini tekshiring; "
                "keyin shu buyruqni qayta ishga tushiring."
            ) from error
        raise
    print(f"Camera index: {camera_index} (requested: {args.camera:g})")

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)
    stable_count = 0
    last_candidate = None
    frame_number = 0
    last_keypoints = None
    last_scores = None
    hand_smoothers = {
        "left": HandLandmarkSmoother(min_score=MIN_KEYPOINT_SCORE),
        "right": HandLandmarkSmoother(min_score=MIN_KEYPOINT_SCORE),
    }
    last_text = ""
    last_color = (0, 165, 255)
    last_time = time.perf_counter()
    fps = 0.0
    print("Kamera ishga tushdi! Dasturni to'xtatish uchun 'q' ni bosing.")

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Dataset 640x480 kadrdan yig'ilgan, shuning uchun inference masshtabini
        # training ma'lumotlari bilan bir xil qilamiz.
        frame = cv2.flip(frame, 1)
        inference_frame = cv2.resize(frame, INFERENCE_SIZE)
        frame_number += 1
        frame_show = inference_frame.copy()
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
        last_time = now

        # Pose modelini har bir kadrda emas, har ikkinchi kadrda ishlatamiz.
        if frame_number % PROCESS_EVERY_N_FRAMES == 0:
            keypoints, scores = openpose(inference_frame)
            last_keypoints = keypoints
            last_scores = scores

        if last_keypoints is not None and len(last_keypoints) > 0:
            keypoints = last_keypoints
            scores = last_scores
            kpts = keypoints[0]
            kpts_scores = scores[0]
            validate_rtmlib_output(kpts, kpts_scores)
            left_state = hand_smoothers["left"].update(
                kpts[91:112], kpts_scores[91:112]
            )
            right_state = hand_smoothers["right"].update(
                kpts[112:133], kpts_scores[112:133]
            )
            candidates = [
                (left_state.keypoints, left_state.scores),
                (right_state.keypoints, right_state.scores),
            ]
            valid_candidates = [
                item for item in candidates
                if item[0] is not None and item[1] is not None
            ]
            hand_kpts, hand_scores = max(
                valid_candidates,
                key=lambda item: float(np.mean(item[1])),
            ) if valid_candidates else (None, None)
                
            if (
                hand_kpts is not None
                and hand_scores is not None
                and float(np.min(hand_scores)) >= MIN_KEYPOINT_SCORE
            ):
                # Ekranga skeletni chizish
                frame_show = draw_skeleton(frame_show, keypoints, scores, kpt_thr=0.2)
                
                # 2-qadam: Koordinatalarni bilakka (0-chi nuqtaga) nisbatan tozalash
                if feature_mode == "bounding_box":
                    features = extract_bounding_box_features(
                        hand_kpts,
                        (inference_frame.shape[1], inference_frame.shape[0]),
                    )
                else:
                    features = extract_wrist_relative_features(hand_kpts)

                # 3-qadam: Bashorat qilish
                probabilities = classifier.predict_proba([features])[0]
                ranking = np.argsort(probabilities)[::-1]
                confidence = float(probabilities[ranking[0]])
                margin = confidence - float(probabilities[ranking[1]])
                candidate = display_label(classifier, int(ranking[0]), args.model)

                if confidence >= CONFIDENCE_THRESHOLD and margin >= MIN_CONFIDENCE_MARGIN:
                    if candidate == last_candidate:
                        stable_count += 1
                    else:
                        last_candidate = candidate
                        stable_count = 1
                    recent_predictions.append(candidate)
                    if stable_count >= MIN_STABLE_FRAMES:
                        stable_prediction = Counter(recent_predictions).most_common(1)[0][0]
                        last_text = f"Harf: {stable_prediction} ({confidence:.0%})"
                    else:
                        last_text = f"Tasdiqlanmoqda: {candidate} ({confidence:.0%})"
                    last_color = (0, 255, 0)
                else:
                    stable_count = 0
                    last_candidate = None
                    last_text = f"Noaniq ({confidence:.0%}, farq {margin:.0%})"
                    last_color = (0, 165, 255)
                top_three = " / ".join(
                    f"{display_label(classifier, int(index), args.model)}:{probabilities[index]:.0%}"
                    for index in ranking[:3]
                )
                cv2.putText(
                    frame_show,
                    top_three,
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            else:
                recent_predictions.clear()
                stable_count = 0
                last_candidate = None
                last_text = "Qo'l aniqlanmadi"
                last_color = (0, 165, 255)
        else:
            recent_predictions.clear()
            stable_count = 0
            last_candidate = None
            last_text = "Qo'l aniqlanmadi"
            last_color = (0, 165, 255)

        if last_text:
            cv2.putText(
                frame_show,
                last_text,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                last_color,
                3,
            )
        cv2.putText(
            frame_show,
            f"FPS: {fps:.1f} | Harf rejimi",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow('Sign Language AI - Live', frame_show)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()