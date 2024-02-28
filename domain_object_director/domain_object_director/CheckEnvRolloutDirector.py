from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class CheckEnvRolloutDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_env(env_name)
        builder.build_config_xml_generation(env_name)
        # # --- after config ---
        builder.build_original_xml_path()
        builder.build_env_instance()
        builder.build_task_space(env_name, mode="torch")
        return builder.get_domain_object()
