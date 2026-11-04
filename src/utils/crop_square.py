import cv2
import numpy as np


def crop_square(jpeg: bytes, sensor_orientation: int) -> np.ndarray | None:
    _ROTATIONS = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    rotation = _ROTATIONS.get(sensor_orientation % 360)
    if rotation is not None:
        frame = cv2.rotate(frame, rotation)
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = frame[top : top + side, left : left + side]
    return square
