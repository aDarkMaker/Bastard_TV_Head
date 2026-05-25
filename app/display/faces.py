import numpy as np  # pyright: ignore[reportMissingImports]

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful"]
MOUTH_LEVELS = {0: "closed", 1: "half", 2: "open"}

GRID_W = 32
GRID_H = 28

PIXEL_OFF = 0
PIXEL_EYE = 1
PIXEL_MOUTH = 2


def _shift(coords, dx, dy):
    return {(x + dx, y + dy) for x, y in coords}


def _eyes_neutral():
    left = {(1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (2, 2), (3, 2), (4, 2)}
    return _shift(left, 8, 9) | _shift(left, 19, 9)


def _eyes_happy():
    left = {(1, 2), (2, 1), (3, 1), (4, 2)}
    return _shift(left, 8, 10) | _shift(left, 19, 10)


def _eyes_sad():
    left = {(1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (4, 2), (2, 3), (3, 3)}
    return _shift(left, 8, 9) | _shift(left, 19, 9)


def _eyes_angry():
    left = {(1, 1), (2, 2), (3, 2), (4, 1), (2, 3), (3, 3)}
    right = {(1, 2), (2, 1), (3, 1), (4, 2), (2, 3), (3, 3)}
    return _shift(left, 8, 9) | _shift(right, 19, 9)


def _eyes_surprised():
    left = {(2, 0), (1, 1), (3, 1), (1, 2), (3, 2), (2, 3)}
    return _shift(left, 8, 9) | _shift(left, 19, 9)


def _eyes_fearful():
    left = {(2, 1), (1, 2), (3, 2), (2, 3)}
    return _shift(left, 9, 10) | _shift(left, 20, 10)


def _mouth_line():
    return {(x, 2) for x in range(3, 8)}


def _mouth_triangle_small():
    return {(x, 2) for x in range(3, 7)} | {(2, 3), (7, 3)} | {(x, 4) for x in range(3, 7)}


def _mouth_triangle_big():
    return (
        {(x, 1) for x in range(3, 7)}
        | {(2, y) for y in range(2, 5)}
        | {(7, y) for y in range(2, 5)}
        | {(x, 5) for x in range(3, 7)}
    )


def _eye_coords(emotion):
    fn = {
        "neutral": _eyes_neutral,
        "happy": _eyes_happy,
        "sad": _eyes_sad,
        "angry": _eyes_angry,
        "surprised": _eyes_surprised,
        "fearful": _eyes_fearful,
    }
    return fn.get(emotion, _eyes_neutral)()


def _mouth_coords(level):
    if level == 0:
        return _mouth_line()
    if level == 1:
        return _mouth_triangle_small()
    return _mouth_triangle_big()


def get_face_grid(emotion, mouth_level):
    grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)

    for x, y in _eye_coords(emotion):
        grid[y, x] = PIXEL_EYE

    mouth_base = (11, 17)
    for x, y in _mouth_coords(mouth_level):
        grid[y + mouth_base[1], x + mouth_base[0]] = PIXEL_MOUTH

    return grid
