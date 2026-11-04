import cv2
import numpy as np


def encode_jpeg(frame: np.ndarray) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return encoded.tobytes() if ok else None
