"""Constants copied from the public checkpoint's preprocessing contract."""

NOSE = [1, 2, 98, 327]
LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181,
    84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311,
    310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LHAND = list(range(522, 543))
RHAND = list(range(501, 522))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

if len(POINT_LANDMARKS) != 118 or 17 not in POINT_LANDMARKS:
    raise RuntimeError("Public Uzbek checkpoint landmark schema is invalid.")
