from domain_object_builder.filtering_model import FilteringModelDomainObject, FilteringModelDomainObjectBuilder
from ..AbstractDomainObjectDirector import AbstractDomainObjectDirector
from domain_object_builder.model import ModelDomainObject


class FilteringMoelDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(
            builder             : FilteringModelDomainObjectBuilder,
            model_domain_object : ModelDomainObject
        ) -> FilteringModelDomainObject:

        # ---- config ----
        builder.build_config_loader()
        builder.build_config()
        # ---- After config ----
        builder.build_filtering(model_domain_object)
        return builder.get_domain_object()
