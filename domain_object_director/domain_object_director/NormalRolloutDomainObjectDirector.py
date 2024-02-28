from domain_object_builder import DomainObject, DomainObjectBuilder
from .AbstractDomainObjectDirector import AbstractDomainObjectDirector


class NormalRolloutDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: DomainObjectBuilder) -> DomainObject:
        builder.build_config()
        builder.build_env_object()
        builder.build_task_space_diff()
        return builder.get_domain_object()
