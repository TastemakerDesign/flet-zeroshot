from pathlib import Path

import onnxruntime as ort

_COREML_OPTIONS = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}
# CoreML can't compile the CLIP models with a dynamic batch dimension, so pin it.
_BATCH_SIZE = 1


def create_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_free_dimension_override_by_name("batch_size", _BATCH_SIZE)
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), options, providers=providers)


def available_providers() -> list[str]:
    return ort.get_available_providers()
