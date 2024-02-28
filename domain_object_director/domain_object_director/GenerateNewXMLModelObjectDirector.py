from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class GenerateNewXMLModelObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_cdsvae_test()
        builder.build_config_env(env_name)
        builder.build_config_icem_sub(env_name)
        builder.build_config_eval()
        builder.build_config_reference(env_name)
        builder.build_config_xml_generation()
        # --- after config ---
        builder.build_xml_generator()
        return builder.get_domain_object()
