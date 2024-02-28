from domain_object_builder import DomainObject, DomainObjectBuilder
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class iCEMRolloutDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_cdsvae_test()
        builder.build_config_env(env_name)
        builder.build_config_icem_sub(env_name)
        builder.build_config_eval()
        builder.build_config_reference(env_name)
        # --- after config ---
        builder.build_env_object()
        builder.build_task_space_diff()
        builder.build_env_data_container()
        builder.build_env_data_repository(dataset_dir=None, read_only=False)
        builder.build_reference()
        builder.build_cost_env()
        builder.build_control_adaptor()
        builder.build_env_planning()
        builder.build_population_sampler()
        builder.build_icem_multiprocessing()
        builder.build_icem_subparticle_manager()
        return builder.get_domain_object()
