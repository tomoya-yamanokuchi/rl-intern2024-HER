from pprint import pprint
from cdsvae.custom import load_config
from .ModelSetParamsDict import ModelSetParamsDict


class ModelMetadataLoader:
    def __init__(self, config_eval):
        self.config_eval = config_eval

    def load(self, model, load_ensemble: bool = True):
        config_model,    model_dir    = self._load_model_config(model)
        # ---
        if load_ensemble:
            config_ensemble, ensemble_dir = self._load_ensemble_config(model_dir)
        else:
            config_ensemble = None; ensemble_dir = None
        # ---
        return ModelSetParamsDict(
            config_model    = config_model,
            model_dir       = model_dir,
            config_ensemble = config_ensemble,
            ensemble_dir    = ensemble_dir,
        )

    def _load_model_config(self, model):
        return load_config(
            log_dir     = self.config_eval.model_set_load_dir,
            group_name  = self.config_eval.model_set.model.group,
            checkpoints = self.config_eval.model_set.model.checkpoints,
            model_name  = model,
        )

    def _load_ensemble_config(self, model_dir: str):
        return load_config(
            log_dir     = model_dir,
            group_name  = "logs_ensemble",
            model_name  = "ensemble",
            checkpoints = self.config_eval.model_set.ensemble.checkpoints,
        )
