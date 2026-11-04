import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from utils.ClipClassifier import ClipClassifier
from utils.create_session import create_session
from utils.ModelPrecision import ModelPrecision
from utils.MovingAverage import MovingAverage

_MODEL_DIR = Path(__file__).parent.parent / "models" / "fastsam"
_MODELS = {
    ModelPrecision.FLOAT32: "FastSAM-s.onnx",
    ModelPrecision.FLOAT16: "FastSAM-s_fp16.onnx",
}
_INPUT_SIZE = 640
_INFERENCE_WINDOW = 5
_MIN_CONFIDENCE = 0.4
_MAX_IOU = 0.9
_MAX_SEGMENTS = 32
_MIN_MASK_AREA = 100
_BORDER_SNAP = 20


@dataclass
class Detection:
    left: float
    top: float
    right: float
    bottom: float
    similarity: float
    baseline_similarity: float | None

    def is_match(self, threshold: float) -> bool:
        if self.similarity < threshold:
            return False
        return self.baseline_similarity is None or self.similarity > self.baseline_similarity


class FastSamDetector:
    def __init__(self, precision: ModelPrecision):
        self._sam = create_session(_MODEL_DIR / _MODELS[precision])
        self._clip = ClipClassifier(precision)
        self._inference_time = MovingAverage(_INFERENCE_WINDOW)

    def encode_text(self, prompt: str) -> np.ndarray:
        return self._clip.encode_text(prompt)

    def inference_times(self) -> tuple[float | None, float | None]:
        return self._inference_time.value(), self._clip.inference_time.value()

    def detect(
        self, frame: np.ndarray, text_embedding: np.ndarray, baseline_embedding: np.ndarray | None
    ) -> Detection | None:
        crops = []
        boxes = []
        for box, mask in self._segment(frame):
            left, top, right, bottom = box
            mask = mask[top:bottom, left:right]
            if mask.sum() <= _MIN_MASK_AREA:
                continue
            crops.append(_pad_square(frame[top:bottom, left:right] * mask[..., None]))
            boxes.append(box)
        if not crops:
            return None
        image_embeddings = self._clip.encode_images(crops)
        similarities = image_embeddings @ text_embedding
        baselines = None if baseline_embedding is None else image_embeddings @ baseline_embedding
        best = _best_segment(similarities, baselines)
        height, width = frame.shape[:2]
        left, top, right, bottom = boxes[best]
        return Detection(
            left / width,
            top / height,
            right / width,
            bottom / height,
            float(similarities[best]),
            None if baselines is None else float(baselines[best]),
        )

    def _segment(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
        height, width = frame.shape[:2]
        pixels = cv2.cvtColor(cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE)), cv2.COLOR_BGR2RGB)
        pixels = pixels.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        started_at = time.perf_counter()
        outputs = self._sam.run(None, {"images": pixels})
        self._inference_time.add(time.perf_counter() - started_at)
        predictions, protos = (cast(np.ndarray, output)[0] for output in outputs)
        predictions = predictions.T
        predictions = predictions[predictions[:, 4] > _MIN_CONFIDENCE]
        if not len(predictions):
            return []
        center_x, center_y, box_width, box_height = predictions[:, :4].T
        boxes = np.stack(
            [
                center_x - box_width / 2.0,
                center_y - box_height / 2.0,
                center_x + box_width / 2.0,
                center_y + box_height / 2.0,
            ],
            axis=1,
        )
        keep = _nms(boxes, predictions[:, 4], _MAX_IOU)[:_MAX_SEGMENTS]
        boxes = boxes[keep]
        channels, proto_height, proto_width = protos.shape
        logits = (predictions[keep, 5:] @ protos.reshape(channels, -1)).reshape(-1, proto_height, proto_width)
        xs = np.arange(_INPUT_SIZE)[None, :]
        ys = np.arange(_INPUT_SIZE)[:, None]
        masks = [
            (
                (cv2.resize(mask, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR) > 0.0)
                & (xs >= box[0])
                & (xs < box[2])
                & (ys >= box[1])
                & (ys < box[3])
            ).astype(np.float32)
            for mask, box in zip(logits.astype(np.float32), boxes)
        ]
        size = np.array([width, height, width, height])
        pixel_boxes = np.clip(boxes * size / _INPUT_SIZE, 0, size)
        near_start = pixel_boxes[:, :2] < _BORDER_SNAP
        pixel_boxes[:, :2][near_start] = 0
        near_end = pixel_boxes[:, 2:] > size[2:] - _BORDER_SNAP
        pixel_boxes[:, 2:] = np.where(near_end, size[2:], pixel_boxes[:, 2:])
        return [
            (
                cast(tuple[int, int, int, int], tuple(int(v) for v in box)),
                cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) > 0.5,
            )
            for box, mask in zip(pixel_boxes, masks)
        ]


# Picks the segment with the highest score for either query, so swapping the two queries selects the same segment.
def _best_segment(similarities: np.ndarray, baselines: np.ndarray | None) -> int:
    if baselines is None:
        return int(similarities.argmax())
    return int(np.maximum(similarities, baselines).argmax())


def _nms(boxes: np.ndarray, scores: np.ndarray, max_iou: float) -> np.ndarray:
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        best, rest = order[0], order[1:]
        keep.append(best)
        overlap_width = np.minimum(boxes[best, 2], boxes[rest, 2]) - np.maximum(boxes[best, 0], boxes[rest, 0])
        overlap_height = np.minimum(boxes[best, 3], boxes[rest, 3]) - np.maximum(boxes[best, 1], boxes[rest, 1])
        intersection = np.clip(overlap_width, 0.0, None) * np.clip(overlap_height, 0.0, None)
        order = rest[intersection / (areas[best] + areas[rest] - intersection) <= max_iou]
    return np.array(keep, dtype=int)


def _pad_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    left = (side - width) // 2
    return cv2.copyMakeBorder(
        image, top, side - height - top, left, side - width - left, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
