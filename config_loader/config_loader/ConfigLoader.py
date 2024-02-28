import os
from copy import deepcopy
from .hydra_compose import hydra_compose
from .EnvNameObject import EnvNameObject
from service import generate_random_string


class ConfigLoader:
    def load_test_datamodule(self):
        config_model = hydra_compose(
            config_path = "../../cdsvae/conf/datamodule",
            config_name = "test",
            env_name    = None,
        )
        return config_model

    def load_model(self):
        # ---
        config_model = hydra_compose(
            config_path = "../../cdsvae/conf",
            config_name = "config",
            env_name    = None,
        )
        return config_model

    def load_ensemble(self):
        config_model = hydra_compose(
            config_path = "../../cdsvae/conf",
            config_name = "config_ensemble",
            env_name    = None,
        )
        return config_model

    def load_env(self, env_name: str):
        envName = EnvNameObject(env_name)
        return hydra_compose(config_path="../../robel_dclaw_env/conf", config_name="config", env_name=envName.env_name)

    def load_icem_single(self, env_name: str):
        envName = EnvNameObject(env_name)
        return hydra_compose(
            config_path = "../../icem_torch/conf",
            config_name = "config_task_space_single_particle_{}".format(envName.env_name_without_instance),
            env_name    = None
        )

    def load_icem_sub(self, env_name: str):
        envName = EnvNameObject(env_name)
        return hydra_compose(
            config_path = "../../icem_torch/conf",
            config_name = "config_task_space_sub_particle_{}".format(envName.env_name_without_instance),
            env_name    = None,
        )

    def load_eval(self):
        config_eval = hydra_compose(
            config_path = "../../config_loader/config_loader/conf",
            config_name = "config_eval",
            env_name    = None,
        )
        if config_eval.tag is None:
            config_eval.tag = generate_random_string()
        return config_eval

    def load_reference(self, env_name: str):
        envName = EnvNameObject(env_name)
        return hydra_compose(
            config_path = "../../reference/conf",
            config_name = "config_{}".format(envName.env_name_without_instance),
            env_name    = envName.env_name_without_instance,
        )

    def load_xml_generation(self, env_name: str):
        envName = EnvNameObject(env_name)
        return hydra_compose(
            config_path = "../../xml_generation/conf",
            config_name = "config_{}".format(envName.env_name_without_instance),
            env_name    = None,
        )



