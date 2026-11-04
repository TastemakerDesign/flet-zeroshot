import flet_camera as fc

from utils.ModelPrecision import ModelPrecision
from utils.RecordingsDB import RecordingsDB, recordings_db

RESOLUTIONS = {
    fc.ResolutionPreset.LOW: "Low",
    fc.ResolutionPreset.MEDIUM: "Medium",
    fc.ResolutionPreset.HIGH: "High",
}
MODEL_PRECISIONS = {
    ModelPrecision.FLOAT16: "Float16",
    ModelPrecision.FLOAT32: "Float32",
}


class SettingsManager:
    _RESOLUTION_SETTING = "resolution"
    _DEFAULT_RESOLUTION = fc.ResolutionPreset.LOW
    _VIBRATION_SETTING = "vibration"
    _ONBOARDED_SETTING = "onboarded"
    _INFERENCE_OVERLAY_SETTING = "inference_overlay"
    _MODEL_PRECISION_SETTING = "model_precision"
    _DEFAULT_MODEL_PRECISION = ModelPrecision.FLOAT16

    def __init__(self, db: RecordingsDB):
        self._db = db

    def get_resolution(self) -> fc.ResolutionPreset:
        value = self._db.get_setting(self._RESOLUTION_SETTING, self._DEFAULT_RESOLUTION.value)
        try:
            resolution = fc.ResolutionPreset(value)
        except ValueError:
            return self._DEFAULT_RESOLUTION
        return resolution if resolution in RESOLUTIONS else self._DEFAULT_RESOLUTION

    def set_resolution(self, resolution: fc.ResolutionPreset) -> None:
        self._db.set_setting(self._RESOLUTION_SETTING, resolution.value)

    def get_vibration(self) -> bool:
        return self._db.get_setting(self._VIBRATION_SETTING, "1") == "1"

    def set_vibration(self, enabled: bool) -> None:
        self._db.set_setting(self._VIBRATION_SETTING, "1" if enabled else "0")

    def get_model_precision(self) -> ModelPrecision:
        value = self._db.get_setting(self._MODEL_PRECISION_SETTING, self._DEFAULT_MODEL_PRECISION.value)
        try:
            return ModelPrecision(value)
        except ValueError:
            return self._DEFAULT_MODEL_PRECISION

    def set_model_precision(self, precision: ModelPrecision) -> None:
        self._db.set_setting(self._MODEL_PRECISION_SETTING, precision.value)

    def get_inference_overlay(self) -> bool:
        return self._db.get_setting(self._INFERENCE_OVERLAY_SETTING, "0") == "1"

    def set_inference_overlay(self, enabled: bool) -> None:
        self._db.set_setting(self._INFERENCE_OVERLAY_SETTING, "1" if enabled else "0")

    def get_onboarded(self) -> bool:
        try:
            return self._db.get_setting(self._ONBOARDED_SETTING, "0") == "1"
        except Exception:
            return False

    def set_onboarded(self) -> None:
        self._db.set_setting(self._ONBOARDED_SETTING, "1")


settings_manager = SettingsManager(recordings_db)
