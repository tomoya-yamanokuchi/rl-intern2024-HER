from torch import nn
from omegaconf import DictConfig


class ModelDomainObject:
    def __init__(self):
        self.config_icem = None
        import ipdb; ipdb.set_trace()

    # ---- config ----
    def set_config_loader(self, configLoader):
        from config_loader import ConfigLoader
        self.configLoader : ConfigLoader = configLoader

    def set_config(self, config: DictConfig):
        self.config = config

    # ---- encoder ----
    def set_frame_encoder(self, frame_encoder):
        from cdsvae.domain.model.inference_model.frame_encoder import AbstractFrameEncoder
        self.frame_encoder : AbstractFrameEncoder = frame_encoder

    def set_forward_lstm_encoder(self, forward_lstm_encoder):
        from cdsvae.domain.model.inference_model.recurrent_encoder import AbstractLSTMEncoder
        self.forward_lstm_encoder : AbstractLSTMEncoder = forward_lstm_encoder

    def set_backward_lstm_encoder(self, backward_lstm_encoder):
        from cdsvae.domain.model.inference_model.recurrent_encoder import AbstractLSTMEncoder
        self.backward_lstm_encoder : AbstractLSTMEncoder = backward_lstm_encoder

    def set_content_encoder(self, content_encoder: nn.Module):
        from cdsvae.domain.model.inference_model.content_encoder import AbstractContentEncoder
        self.content_encoder : AbstractContentEncoder = content_encoder

    def set_motion_encoder(self, motion_encoder):
        from cdsvae.domain.model.inference_model.motion_encoder import AbstractMotionEncoder
        self.motion_encoder : AbstractMotionEncoder = motion_encoder


    # ---- linear cast for common transformation in model ----
    def set_linear_cast_f_trans(self, linear_cast_f_trans: nn.Module):
        self.linear_cast_f_trans = linear_cast_f_trans

    def set_linear_cast_f_emiss(self, linear_cast_f_emiss: nn.Module):
        self.linear_cast_f_emiss = linear_cast_f_emiss

    def set_linear_cast_z(self, linear_cast_z: nn.Module):
        self.linear_cast_z = linear_cast_z

    def set_linear_cast_u(self, linear_cast_u: nn.Module):
        self.linear_cast_u = linear_cast_u

    # ---- prior ----
    def set_content_prior(self, content_prior):
        from cdsvae.domain.model.generative_model import ContentPrior
        self.content_prior : ContentPrior = content_prior

    def set_motion_prior(self, motion_prior):
        from cdsvae.domain.model.generative_model.motion_prior import AbstracMotionPrior
        self.motion_prior : AbstracMotionPrior = motion_prior

    # ---- decoder ----
    def set_frame_decoder(self, frame_decoder: nn.Module):
        from cdsvae.domain.model.generative_model.frame_decoder import AbstractFrameDecoder
        self.frame_decoder : AbstractFrameDecoder = frame_decoder

    def set_sensor_robot_decoder(self, sensor_robot_decoder: nn.Module):
        self.sensor_robot_decoder = sensor_robot_decoder

    def set_sensor_object_decoder(self, sensor_object_decoder: nn.Module):
        self.sensor_object_decoder = sensor_object_decoder
