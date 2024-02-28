from service import join_with_mkdir
from domain_object_builder import DomainObject, DomainObjectBuilder
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class LoadEnsembleDatasetDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder, dataset_dir : str) -> DomainObject:
        # --- config ----
        builder.build_config_loader()
        builder.build_config_model()
        # ---- after config ----
        builder.build_ensemble_data_container()
        builder.build_shelve_repository(save_dir=None, read_only=True)
        builder.build_ensemble_data_stack()
        builder.build_ensemble_data_stack_info_writer()
        builder.build_ensemble_dataset_visualizer(save_dir=join_with_mkdir(dataset_dir, "figs", is_end_file=False))
        return builder.get_domain_object()
