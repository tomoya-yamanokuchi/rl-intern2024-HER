from torch import nn
from omegaconf import DictConfig
from .ViRCCEDomainObject import ViRCCEDomainObject


class ViRCCEDomainObjectBuilder:
    def __init__(self, config):
        self.domain_object = ViRCCEDomainObject()
        self.domain_object.set_config(config)
        self._set_common_params()

    def _set_common_params(self):
        # import ipdb; ipdb.set_trace()
        self.dim_state   = self.domain_object.config.model.motion.dim_params.dim_state
        self.dim_ctrl    = self.domain_object.config.model.motion.dim_params.dim_ctrl
        self.dim_content = self.domain_object.config.model.content.dim_content.dim_content
        self.dim_a       = self.domain_object.config.model.frame_encoder.dim_a
        self.channel     = self.domain_object.config.model.frame_encoder.in_channels

    # ---- linear cast for common transformation in model ----
    def build_linear_cast_z(self):
        linear_cast_z = nn.Linear(self.dim_state, self.dim_state)
        self.domain_object.set_linear_cast_z(linear_cast_z)

    def build_linear_cast_u(self):
        linear_cast_u = nn.Linear(self.dim_ctrl, self.dim_state)
        self.domain_object.set_linear_cast_u(linear_cast_u)

    # ---- encoder ----
    def build_frame_encoder(self):
        from cdsvae.domain.model.inference_model.frame_encoder import FrameEncoderFactory
        frame_encoder = FrameEncoderFactory.create(**self.domain_object.config.model.frame_encoder)
        self.domain_object.set_frame_encoder(frame_encoder)

    def build_content_encoder(self):
        from cdsvae.domain.model.inference_model.content_encoder import ContentEncoderFactory
        # import ipdb; ipdb.set_trace()
        content_encoder = ContentEncoderFactory.create(
            dim_frame_feature = self.dim_a,
            dim_content       = self.dim_content,
            dim_state         = self.dim_state,
            **self.domain_object.config.model.content.encoder,
        )
        self.domain_object.set_content_encoder(content_encoder)

    def build_motion_encoder(self):
        from cdsvae.domain.model.inference_model.motion_encoder import MotionEncoderFactory
        motion_encoder = MotionEncoderFactory().create(
            dim_a         = self.dim_a,
            dim_sensor    = self.domain_object.config.model.sensor_robot.dim_sensor,
            linear_cast_z = self.domain_object.linear_cast_z,
            linear_cast_u = self.domain_object.linear_cast_u,
            **self.domain_object.config.model.motion.encoder,
            **self.domain_object.config.model.motion.dim_params,
        )
        self.domain_object.set_motion_encoder(motion_encoder)

    # ---- prior ----
    def build_content_prior(self):
        from cdsvae.domain.model.generative_model.content_prior import ContentPriorFactory
        # import ipdb; ipdb.set_trace()
        content_prior = ContentPriorFactory().create(
            name          = self.domain_object.config.model.content.prior.name,
            context_dim   = self.dim_content,
            num_keyframes = self.domain_object.config.model.content.encoder.num_keyframes,
        )
        self.domain_object.set_content_prior(content_prior)

    def build_motion_prior(self):
        from cdsvae.domain.model.generative_model.motion_prior import MotionPriorFactory
        motion_prior = MotionPriorFactory.create(
            dim_content         = self.dim_content,
            linear_cast_z       = self.domain_object.linear_cast_z,
            linear_cast_u       = self.domain_object.linear_cast_u,
            num_keyframes       = self.domain_object.config.model.content.encoder.num_keyframes,
            **self.domain_object.config.model.motion.prior,
            **self.domain_object.config.model.motion.dim_params,
        )
        self.domain_object.set_motion_prior(motion_prior)

    # ---- decoder ----
    def build_latent_frame_decoder(self):
        from cdsvae.domain.model.generative_model.latent_frame_decoder import LatentFrameDecoderFactory
        latent_frame_decoder = LatentFrameDecoderFactory().create(
            **self.domain_object.config.model.latent_frame_decoder,
            dim_content         = self.dim_content,
            dim_state           = self.dim_state,
            dim_a               = self.dim_a,
            num_keyframes       = self.domain_object.config.model.content.encoder.num_keyframes,
        )
        self.domain_object.set_latent_frame_decoder(latent_frame_decoder)

    def build_frame_decoder(self):
        from cdsvae.domain.model.generative_model.frame_decoder import FrameDecoderFactory
        frame_decoder = FrameDecoderFactory().create(
            **self.domain_object.config.model.frame_decoder,
            in_dim       = self.dim_a,
            out_channels = self.channel,
        )
        self.domain_object.set_frame_decoder(frame_decoder)

    def build_sensor_robot_decoder(self):
        from cdsvae.domain.model.generative_model.sensor_decoder import SensorDecoderFactory
        sensor_robot_decoder  = SensorDecoderFactory().create(
            **self.domain_object.config.model.sensor_robot,
            dim_state = self.dim_state
        )
        self.domain_object.set_sensor_robot_decoder(sensor_robot_decoder)

    def build_sensor_object_decoder(self):
        from cdsvae.domain.model.generative_model.sensor_decoder import SensorDecoderFactory
        sensor_object_decoder  = SensorDecoderFactory().create(
            **self.domain_object.config.model.sensor_object,
            dim_state = self.dim_state
        )
        self.domain_object.set_sensor_object_decoder(sensor_object_decoder)

    def get_domain_object(self) -> ViRCCEDomainObject:
        return self.domain_object

