from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class RandomCtrlDataCollectionObjectDirectorWithFixedInitialMotion(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, env_name: str, dataset_name : str= None) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_env(env_name)
        builder.build_config_icem_sub(env_name)
        builder.build_config_reference(env_name)
        builder.build_config_xml_generation(env_name)
        # --- after config ---
        # builder.build_task_space(env_name, mode="numpy")
        builder.build_task_space(env_name, mode="torch")
        builder.build_trajectory_evaluator()
        builder.build_adapter(env_name)
        builder.build_file_copy_manager(dataset_name)
        builder.build_original_xml_path()
        builder.build_xml_generator()
        builder.build_env_object()
        builder.build_env_data_container()
        builder.build_env_data_repository(dataset_dir=None, read_only=False)
        builder.build_reference() # データ収集自体には必要ないが，iCEM_Managerを動かす上で必要
        builder.build_control_adaptor()
        builder.build_population_sampler()
        builder.build_data_collection_planning_with_fixed_init_ctrl()
        builder.build_icem_multiprocessing_random_data_collection_with_fixed_initial_motion()
        builder.build_icem_subparticle_manager()
        return builder.get_domain_object()
