from domain_object_builder.model.kvae import KVAEDomainObject, KVAEDomainObjectBuilder
from ..AbstractDomainObjectDirector import AbstractDomainObjectDirector


class KVAEDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: KVAEDomainObjectBuilder) -> KVAEDomainObject:
        # ---- vae ----
        builder.build_encoder()
        builder.build_decoder()
        # --- lgssm ---
        builder.build_lgssm_matrix_params()
        builder.build_kalman_filter()
        builder.build_kalman_smoother()
        builder.build_object_state_decoder()
        # ---
        return builder.get_domain_object()
