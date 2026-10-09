
import csv
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils import shuffle


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "gesture_classifier.pkl"
METRICS_PATH = ROOT / "data" / "processed" / "metrics.txt"


def load_dataset(path):
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        if not columns or "label" not in columns:
            raise ValueError(f"{path} faylida label ustuni yo'q.")

        feature_columns = [c for c in columns if c != "label"]
        X, y = [], []

        for row in reader:
            X.append([float(row[c]) for c in feature_columns])
            y.append(row["label"])

    if not X:
        raise ValueError(f"{path} bo'sh.")

    return X, y, feature_columns


def main():
    X_train, y_train, feature_columns = load_dataset(TRAIN_PATH)
    X_test, y_test, test_columns = load_dataset(TEST_PATH)

    if feature_columns != test_columns:
        raise ValueError("Train va test feature ustunlari mos emas.")

    if set(y_test) - set(y_train):
        raise ValueError("Testda train tarkibida bo'lmagan klass bor.")

    # Train ma'lumotlarini aralashtirish natijani takrorlashga imkon beradi.
    X_train, y_train = shuffle(
        X_train, y_train, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("Training boshlandi...")
    model.fit(X_train, y_train)
    print("Training tugadi.")

    # Test faqat yakuniy baholash uchun ishlatiladi.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=sorted(set(y_train) | set(y_test)),
        zero_division=0,
    )
    labels = sorted(set(y_train) | set(y_test))
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "labels": labels,
                "random_state": 42,
                "keypoint_layout": "rtmlib_mmpose_wholebody_v1",
            },
            file,
        )

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        file.write(f"Accuracy: {accuracy:.4%}\n\n")
        file.write("Classification report:\n")
        file.write(report)
        file.write("\n\nLabels:\n")
        file.write(", ".join(labels))
        file.write("\n\nConfusion matrix:\n")
        file.write(str(matrix))

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)
    print(f"\nModel saqlandi: {MODEL_PATH}")
    print(f"Metrikalar saqlandi: {METRICS_PATH}")


if __name__ == "__main__":
    main()
