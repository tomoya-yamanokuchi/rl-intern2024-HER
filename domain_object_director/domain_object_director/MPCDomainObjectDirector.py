from domain_object_builder import DomainObject, DomainObjectBuilder, ModelSetParamsDict
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector
from typing import TypedDict
from service import convert_target_to_string


class MPCDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(
                builder             : DomainObjectBuilder,
                env_name            : str,
                metadata            : ModelSetParamsDict,
                paramsDictReference : TypedDict = None
            ) -> DomainObject:
        # --- config ---
        builder.build_config_loader()
        builder.build_config_model()
        builder.build_config_env(env_name)
        builder.build_config_reference(env_name)
        builder.build_config_icem_single(env_name)
        builder.build_config_xml_generation(env_name)
        # builder.build_config_ensemble()
        # --- model ---
        builder.build_task_space(env_name, mode="torch")
        builder.build_trajectory_evaluator(env_name)
        builder.build_adapter(env_name)
        # builder.build_ensemble_adapter(model_dir=metadata["model_dir"])
        builder.build_model_domain_object(config=metadata["config_model"])
        builder.build_model()
        builder.build_lit_model_eval(metadata["config_model"])
        builder.build_filtering_model()
        builder.build_prediction_model()
        builder.build_filtering_manager()
        builder.build_prediction_manager()
        # ---ensemble ---
        # builder.build_ensemble(config=metadata["config_ensemble"])
        # builder.build_lit_ensemble_eval(metadata["config_ensemble"])
        # ---
        builder.build_file_copy_manager()
        builder.build_original_xml_path()
        builder.build_xml_model_modifier()
        builder.build_xml_generator()
        builder.build_env_adapter()
        # ---
        builder.build_task_space(env_name, mode="torch")
        # ---
        # import ipdb; ipdb.set_trace()
        builder.domain_object.config_eval.tag = builder.domain_object.config_eval.tag + "_target={}".format(convert_target_to_string(paramsDictReference["target_position"]))

        print("tag = ", builder.domain_object.config_eval.tag)
        builder.build_icem_repository(model_dir=metadata["model_dir"])
        # ---
        builder.build_reference(paramsDictReference)
        builder.build_planning_visualizer()
        builder.build_model_planning()
        builder.build_cost_model()
        builder.build_icem()
        builder.build_icem_manager()
        builder.build_env_instance()
        builder.build_usecase_repository()
        builder.build_mpc_result_visualizer()
        builder.build_mpc_data_logger()
        builder.build_replay()
        builder.build_image_viewer()
        return builder.get_domain_object()
