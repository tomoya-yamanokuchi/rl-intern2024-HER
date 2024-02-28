from domain_object_builder.model import ViRCCEDomainObject, ViRCCEDomainObjectBuilder
from ..AbstractDomainObjectDirector import AbstractDomainObjectDirector


class ViRCCEDomainObjectDirector(AbstractDomainObjectDirector):
    @staticmethod
    def construct(builder: ViRCCEDomainObjectBuilder) -> ViRCCEDomainObject:
        # --- linear cast ---
        builder.build_linear_cast_z()
        builder.build_linear_cast_u()
        # ---- encoder ----
        builder.build_frame_encoder()
        builder.build_content_encoder()
        builder.build_motion_encoder()
        # ---- prior ----
        builder.build_content_prior()
        builder.build_motion_prior()
        # ---- decoder ----
        builder.build_latent_frame_decoder()
        builder.build_frame_decoder()
        builder.build_sensor_robot_decoder()
        builder.build_sensor_object_decoder()
        # ---
        return builder.get_domain_object()
