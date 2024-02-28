from typing import TypedDict
from omegaconf import DictConfig


class ModelSetParamsDict(TypedDict):
    # ---- model ----
    config_model    : DictConfig
    model_dir       : str
    # --- ensemble ---
    config_ensemble : DictConfig
    ensemble_dir    : str

