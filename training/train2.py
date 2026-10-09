"""
Mustahkam (robust) trening - v2
================================
Joylashtirish:  training/train_v2.py

Ishlatish (loyiha ildizidan):
    python training/train_v2.py
    python training/train_v2.py --no-mirror        # ko'zgu augmentatsiyasisiz
    python training/train_v2.py --folds 5 --copies 2

Nima qiladi:
  1) data/dataset.csv ni o'qiydi (label, x0,y0 ... x20,y20 - bitta qo'l, wrist = (0,0))
  2) "Tasodifiy split" bahosi nega yolg'on ekanini ko'rsatadi
     (uni vaqt bo'yicha blokli baholash bilan solishtiradi)
  3) Jonli sharoitni taqlid qiladi: masofa, burilish, ko'zgu, shovqin
  4) Normalizatsiya + augmentatsiya bilan yangi model o'qitib
     models/gesture_classifier_v2.pkl ga saqlaydi

Eski usul  = xom koordinatalar + Random Forest (hozirgi usulingizga o'xshash)
Yangi usul = masshtab-normalizatsiya + augmentatsiya (ko'zgu, burilish,
             cho'zilish, shovqin) + MLP

Agar dataset.csv da "session" ustuni bo'lsa, baholash sessiyalar bo'yicha
ajratiladi (eng halol usul). Bo'lmasa - har harfning ketma-ket qatorlari
5 ta bo'lakka bo'linadi va test bo'lagi atrofidagi qatorlar o'qitishdan
olib tashlanadi (qo'shni egizak kadrlar oqib o'tmasligi uchun).
"""
import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_PATH = ROOT / "archive" / "legacy-models" / "gesture_classifier_v2.pkl"
REPORT_PATH = ROOT / "data" / "processed" / "metrics_v2.txt"
NORM_VERSION = "maxnorm_v1"   # inference/live_v2.py dagi normalize_hand bilan bir xil bo'lishi shart
SEED = 42

KINDS = [
    ("oddiy",   "Oddiy (xuddi shu sessiya sharoiti)"),
    ("uzoq",    "Uzoqroq turish (qo'l 0.7x kichik)"),
    ("yaqin",   "Yaqinroq turish (qo'l 1.4x katta)"),
    ("rot+",    "Qo'l +15 daraja burilgan"),
    ("rot-",    "Qo'l -15 daraja burilgan"),
    ("kozgu",   "Ko'zgu (boshqa qo'l yoki flip)"),
    ("shovqin", "Nuqtalarda shovqin (3%)"),
]


# ----------------------------------------------------------------- ma'lumot
def load_dataset(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} bo'sh")
    labels = np.array([r["label"] for r in rows])
    pts = np.zeros((len(rows), 21, 2), dtype=np.float64)
    for i, r in enumerate(rows):
        for k in range(21):
            pts[i, k, 0] = float(r[f"x{k}"])
            pts[i, k, 1] = float(r[f"y{k}"])
    sessions = np.array([r["session"] for r in rows]) if "session" in rows[0] else None
    return pts, labels, sessions


# ------------------------------------------------------------ normalizatsiya
def normalize_batch(rel):
    """(n,21,2) wristga nisbatan koordinatalar -> (n,40) masshtabdan mustaqil
    belgilar. Masshtab = wristdan eng uzoq nuqtagacha masofa."""
    rel = rel - rel[:, 0:1, :]
    ext = np.linalg.norm(rel, axis=2).max(axis=1)
    ext = np.where(ext < 1e-6, 1.0, ext)
    return (rel[:, 1:, :] / ext[:, None, None]).reshape(len(rel), -1)


def rotate(p, deg):
    th = np.deg2rad(deg)
    c = np.cos(th)[:, None]
    s = np.sin(th)[:, None]
    x, y = p[:, :, 0], p[:, :, 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def augment(rel, y, rng, copies, mirror):
    base, yb = rel, y
    if mirror:
        base = np.concatenate([rel, rel * np.array([-1.0, 1.0])])
        yb = np.concatenate([y, y])
    xs, ys = [base], [yb]
    for _ in range(copies):
        p = base.copy()
        n = len(p)
        p = rotate(p, rng.uniform(-15, 15, n))
        p[:, :, 0] *= (1 + rng.uniform(-0.15, 0.15, n))[:, None]
        p[:, :, 1] *= (1 + rng.uniform(-0.15, 0.15, n))[:, None]
        ext = np.linalg.norm(p, axis=2).max(axis=1)
        p = p + rng.normal(0.0, 1.0, p.shape) * (0.02 * ext)[:, None, None]
        xs.append(p)
        ys.append(yb)
    return np.concatenate(xs), np.concatenate(ys)


def perturb(rel, kind, rng):
    p = rel.copy()
    if kind == "uzoq":
        p *= 0.7
    elif kind == "yaqin":
        p *= 1.4
    elif kind == "rot+":
        p = rotate(p, np.full(len(p), 15.0))
    elif kind == "rot-":
        p = rotate(p, np.full(len(p), -15.0))
    elif kind == "kozgu":
        p[:, :, 0] *= -1
    elif kind == "shovqin":
        ext = np.linalg.norm(p, axis=2).max(axis=1)
        p = p + rng.normal(0.0, 1.0, p.shape) * (0.03 * ext)[:, None, None]
    return p - p[:, 0:1, :]


# ------------------------------------------------------------------- modellar
def make_old_model():
    return RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED)


def make_new_model():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256,
                      learning_rate_init=1e-3, max_iter=100, tol=1e-4,
                      n_iter_no_change=8, random_state=SEED),
    )


# -------------------------------------------------------------------- splitlar
def make_blocked_splits(labels, k, gap):
    n = len(labels)
    classes = np.unique(labels)
    rank = np.zeros(n, dtype=int)
    chunk = np.zeros(n, dtype=int)
    for c in classes:
        idx = np.where(labels == c)[0]
        cnt = len(idx)
        rank[idx] = np.arange(cnt)
        chunk[idx] = np.minimum(k - 1, np.arange(cnt) * k // cnt)
    splits = []
    for f in range(k):
        test = chunk == f
        train = ~test
        for c in classes:
            idx = np.where(labels == c)[0]
            r = rank[idx]
            in_test = chunk[idx] == f
            if not in_test.any():
                continue
            lo, hi = r[in_test].min(), r[in_test].max()
            near = (r >= lo - gap) & (r <= hi + gap) & (~in_test)
            train[idx[near]] = False
        splits.append((np.where(train)[0], np.where(test)[0]))
    return splits


def make_group_splits(groups, k):
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(k, n_groups))
    return list(gkf.split(np.zeros(len(groups)), groups=groups))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--copies", type=int, default=2, help="augmentatsiya nusxalari soni")
    ap.add_argument("--gap", type=int, default=15, help="test bo'lagi atrofida o'qitishdan olib tashlanadigan qatorlar")
    ap.add_argument("--no-mirror", action="store_true", help="ko'zgu augmentatsiyasini o'chirish")
    args = ap.parse_args()
    mirror = not args.no_mirror
    rng = np.random.default_rng(SEED)
    out = []

    def say(text=""):
        print(text)
        out.append(text)

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Topilmadi: {data_path}")
    pts, labels, sessions = load_dataset(data_path)
    n = len(labels)
    classes = sorted(set(labels))

    say("=== 1) MA'LUMOT ===")
    say(f"Namunalar: {n} | Klasslar: {len(classes)} | "
        f"Sessiya ustuni: {'bor' if sessions is not None else 'yoq (blokli split ishlatiladi)'}")

    # Y yo'nalishi: o'rta barmoq asosi (kp9) wristdan yuqorida/pastdami?
    frac_pos = float(np.mean(pts[:, 9, 1] > 0))
    y_multiplier = -1.0 if frac_pos >= 0.5 else 1.0
    if frac_pos >= 0.5:
        say(f"Y yo'nalishi: kp9 namunalarning {frac_pos * 100:.0f}% ida wristdan YUQORIDA (musbat y). "
            "Demak datada y yuqoriga yo'nalgan -> jonli rejimda kamera y'si teskari qilinadi.")
    else:
        say(f"Y yo'nalishi: kp9 namunalarning {(1 - frac_pos) * 100:.0f}% ida manfiy y "
            "(kamera koordinatalari) -> jonli rejimda y o'zgartirilmaydi.")
    if 0.3 < frac_pos < 0.7:
        say("DIQQAT: y yo'nalishi aniq emas (qo'llar turlicha burilgan). Jonli rejimda 'y' tugmasi bilan sinab ko'ring.")

    # Masshtab "yashirin belgi" bo'lib qolganmi? (klasslararo vs klass ichidagi farq)
    size = np.linalg.norm(pts[:, 9, :] - pts[:, 0, :], axis=1)
    means = np.array([size[labels == c].mean() for c in classes])
    within = float(np.mean([size[labels == c].std() for c in classes]))
    say(f"Qo'l o'lchami (wrist->kp9, piksel): klasslar o'rtachasi {means.min():.1f}..{means.max():.1f}, "
        f"klass ichidagi og'ish ~{within:.1f}")
    if means.std() > 2.0 * within:
        say("DIQQAT: klasslar orasidagi o'lcham farqi klass ichidagidan ancha katta -> model o'lchamni "
            "'yashirin belgi' sifatida ishlatayotgan bo'lishi mumkin.")

    # --- splitlar
    if sessions is not None:
        splits = make_group_splits(sessions, args.folds)
    else:
        splits = make_blocked_splits(labels, args.folds, args.gap)

    # --- eski usulning "tasodifiy split" natijasi
    say("\n=== 2) HALOLROQ BAHOLASH ===")
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.2, stratify=labels, random_state=SEED)
    rf = make_old_model().fit(pts[idx_tr].reshape(len(idx_tr), -1), labels[idx_tr])
    acc_random = float(np.mean(rf.predict(pts[idx_te].reshape(len(idx_te), -1)) == labels[idx_te]))
    say(f"Tasodifiy 80/20 split (eski usul, siz ko'rgan turdagi baho): {acc_random * 100:.1f}%")

    # --- CV: eski va yangi usul + stress testlar
    correct_old = {k: 0 for k, _ in KINDS}
    correct_new = {k: 0 for k, _ in KINDS}
    total = {k: 0 for k, _ in KINDS}
    y_true_all, y_pred_all = [], []
    t0 = time.time()
    for fi, (tr, te) in enumerate(splits, 1):
        print(f"  fold {fi}/{len(splits)} ... ({time.time() - t0:.0f}s)", flush=True)
        old = make_old_model().fit(pts[tr].reshape(len(tr), -1), labels[tr])
        xa, ya = augment(pts[tr], labels[tr], rng, args.copies, mirror)
        new = make_new_model().fit(normalize_batch(xa), ya)
        for kind, _ in KINDS:
            pt = perturb(pts[te], kind, rng)
            p_old = old.predict(pt.reshape(len(te), -1))
            p_new = new.predict(normalize_batch(pt))
            correct_old[kind] += int(np.sum(p_old == labels[te]))
            correct_new[kind] += int(np.sum(p_new == labels[te]))
            total[kind] += len(te)
            if kind == "oddiy":
                y_true_all.extend(labels[te])
                y_pred_all.extend(p_new)

    kind_name = "blokli" if sessions is None else "sessiya bo'yicha"
    say(f"{kind_name} {len(splits)}-fold: eski {correct_old['oddiy'] / total['oddiy'] * 100:.1f}% | "
        f"yangi {correct_new['oddiy'] / total['oddiy'] * 100:.1f}%")
    say("(Tasodifiy split bilan blokli baho orasidagi farq = oldingi 93% ning qancha qismi 'yodlash' ekanini ko'rsatadi.)")

    say("\n=== 3) JONLI SHAROITNI TAQLID QILISH (test bo'laklari ustida) ===")
    say(f"{'Sinov':<40}{'Eski':>9}{'Yangi':>9}")
    for kind, title in KINDS:
        a = correct_old[kind] / total[kind] * 100
        b = correct_new[kind] / total[kind] * 100
        say(f"{title:<40}{a:>8.1f}%{b:>8.1f}%")
    say("Eslatma: blokli test hali ham bitta yozuv sessiyasi ichida. Haqiqiy farq masofa/burilish/ko'zgu "
        "qatorlarida ko'rinadi; jonli rejim undan ham qiyinroq (yangi kun, yorug'lik, odam).")

    # --- eng zaif harflar
    f1 = f1_score(y_true_all, y_pred_all, labels=classes, average=None, zero_division=0)
    order = np.argsort(f1)[:6]
    say("\n=== 4) ENG ZAIF HARFLAR (yangi usul) ===")
    say("  " + " | ".join(f"{classes[i]}: {f1[i]:.2f}" for i in order))
    say("  (bular ko'proq va xilma-xilroq data kerak bo'lgan harflar)")

    # --- yakuniy model: butun datada
    say("\n=== 5) YAKUNIY MODEL ===")
    xa, ya = augment(pts, labels, rng, args.copies, mirror)
    final = make_new_model().fit(normalize_batch(xa), ya)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final,
        "classes": list(final.classes_),
        "y_multiplier": y_multiplier,
        "norm_version": NORM_VERSION,
        "mirror_aug": mirror,
        "n_samples": n,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    say(f"Model saqlandi: {MODEL_PATH}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    say(f"Hisobot saqlandi: {REPORT_PATH}")
    say("\nKeyingi qadam:  python inference/live_v2.py")


if __name__ == "__main__":
    main()