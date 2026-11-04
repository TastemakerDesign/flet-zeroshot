import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from tokenizers import Tokenizer

from utils.create_session import create_session
from utils.ModelPrecision import ModelPrecision
from utils.MovingAverage import MovingAverage

_MODEL_DIR = Path(__file__).parent.parent / "models" / "mobileclip2_s0"
_VISUAL_MODELS = {
    ModelPrecision.FLOAT32: "visual.onnx",
    ModelPrecision.FLOAT16: "visual_fp16.onnx",
}
_IMAGE_SIZE = 256
_CONTEXT_LENGTH = 77
_INFERENCE_WINDOW = 10
# Padding token ID
_PAD_ID = 0
# End-of-text token ID
_EOT_ID = 49407


class ClipClassifier:
    def __init__(self, precision: ModelPrecision):
        self._visual = create_session(_MODEL_DIR / _VISUAL_MODELS[precision])
        self._text = create_session(_MODEL_DIR / "text_int8.onnx")
        self._tokenizer = Tokenizer.from_file(str(_MODEL_DIR / "tokenizer.json"))
        self.inference_time = MovingAverage(_INFERENCE_WINDOW)

    def encode_text(self, prompt: str) -> np.ndarray:
        ids = self._tokenizer.encode(prompt).ids
        if len(ids) > _CONTEXT_LENGTH:
            ids = ids[: _CONTEXT_LENGTH - 1] + [_EOT_ID]
        ids += [_PAD_ID] * (_CONTEXT_LENGTH - len(ids))
        (embedding,) = self._text.run(None, {"input_ids": np.array([ids], np.int64)})
        embedding = cast(np.ndarray, embedding)
        return _normalize(embedding[0])

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        embeddings = []
        for image in images:
            pixels = (
                cv2.cvtColor(
                    cv2.resize(image, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_LINEAR),
                    cv2.COLOR_BGR2RGB,
                )
                .transpose(2, 0, 1)[None]
                .astype(np.float32)
                / 255.0
            )
            started_at = time.perf_counter()
            (embedding,) = self._visual.run(None, {"pixel_values": pixels})
            self.inference_time.add(time.perf_counter() - started_at)
            embeddings.append(cast(np.ndarray, embedding)[0])
        return _normalize(np.stack(embeddings))


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)
