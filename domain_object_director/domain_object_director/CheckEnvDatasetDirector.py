from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class CheckEnvDatasetDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, dataset_dir: str) -> DomainObject:
        # --- config ----
        builder.build_config_cdsvae_test()
        builder.build_config_env()
        builder.build_config_icem_sub()
        builder.build_config_eval()
        # # --- after config ---
        builder.build_env_data_logger(dataset_dir=dataset_dir, read_only=True)
        return builder.get_domain_object()
