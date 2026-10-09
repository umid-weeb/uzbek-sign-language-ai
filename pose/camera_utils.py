from __future__ import annotations

import cv2


def list_cameras(max_devices: int = 10) -> list[int]:
    """Return camera indexes that can actually deliver a frame."""
    available: list[int] = []
    for index in range(max_devices):
        camera = cv2.VideoCapture(index)
        ok, _ = camera.read() if camera.isOpened() else (False, None)
        if ok:
            available.append(index)
        camera.release()
    return available


def open_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    # CAP_ANY is important for Continuity Camera: on some macOS/OpenCV
    # versions AVFoundation opens the index but returns no first frame.
    camera = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not camera.isOpened():
        available = list_cameras()
        camera.release()
        raise RuntimeError(
            f"Kamera {index} ochilmadi. Mavjud camera indexlar: {available}. "
            "iPhone Continuity Camera yoqilganini tekshiring."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, _ = camera.read()
    if not ok:
        camera.release()
        available = list_cameras()
        raise RuntimeError(
            f"Kamera {index} ochildi, lekin kadr qaytarmadi. "
            f"Ishlayotgan camera indexlar: {available}. "
            "iPhone'da Continuity Camera yoqilganini va telefon qulfdan "
            "chiqarilganini tekshiring."
        )
    return camera


def resolve_camera_index(requested: float) -> int:
    """Resolve integer indexes and the explicit 0.5 iPhone alias."""
    if requested == 0.5:
        # Continuity Camera can disappear briefly while macOS changes camera
        # ownership. Do not probe every index here; probing can keep the
        # device busy and make the next open fail. On this setup index 1 is
        # the iPhone, while index 0 is the MacBook camera.
        return 1
    if requested < 0 or not requested.is_integer():
        raise ValueError("Camera index butun son bo'lishi kerak yoki iPhone uchun 0.5 ishlatiladi.")
    return int(requested)
