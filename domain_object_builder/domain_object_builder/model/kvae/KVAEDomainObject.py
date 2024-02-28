from torch import nn
from omegaconf import DictConfig


class KVAEDomainObject:
    def __init__(self):
        self.config_icem = None

    # ---- config ----
    def set_config_loader(self, configLoader):
        from config_loader import ConfigLoader
        self.configLoader : ConfigLoader = configLoader

    def set_config(self, config: DictConfig):
        self.config = config

    # ---------------- VAE -----------------
    def set_encoder(self, encoder):
        from cdsvae.domain.model.kvae.network import ConvFCFrameEncoder
        self.encoder : ConvFCFrameEncoder = encoder

    def set_decoder(self, decoder):
        from cdsvae.domain.model.kvae.network import FullConvFrameDecoder
        self.decoder : FullConvFrameDecoder = decoder

    # --------------- LGSSM -----------------
    def set_matrix_params(self, matrix_params):
        from cdsvae.domain.model.kvae.lgssm import MatrixParams
        self.matrix_params : MatrixParams = matrix_params

    def set_kalman_filter(self, kalman_filter):
        from cdsvae.domain.model.kvae.lgssm.kalmanfilter import KalmanFilter
        self.kalman_filter : KalmanFilter = kalman_filter

    def set_kalman_smoother(self, kalman_smoother):
        from cdsvae.domain.model.kvae.lgssm.kalmansmoother import KalmanSmoother
        self.kalman_smoother : KalmanSmoother = kalman_smoother

    # ------------- object_state_decoder --------------
    def set_object_state_decoder(self, object_state_decoder):
        from cdsvae.domain.model.kvae.network import TwoLinearObjectStateDecoder
        self.object_state_decoder : TwoLinearObjectStateDecoder = object_state_decoder
