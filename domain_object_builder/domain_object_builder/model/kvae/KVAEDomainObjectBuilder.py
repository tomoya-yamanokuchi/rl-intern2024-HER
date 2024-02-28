from omegaconf import DictConfig
from .KVAEDomainObject import KVAEDomainObject


class KVAEDomainObjectBuilder:
    def __init__(self, config):
        self.domain_object = KVAEDomainObject()
        self.domain_object.set_config(config)
        self._set_params()

    def _set_params(self):
        self.dim_z = self.domain_object.config.model.matrix_params.dim_z
        self.dim_y = (self.domain_object.config.model.encoder.dim_a) + (self.domain_object.config.model.matrix_params.dim_b)

    # ---- encoder ----
    def build_encoder(self):
        from cdsvae.domain.model.kvae.network import ConvFCFrameEncoder
        encoder = ConvFCFrameEncoder(**self.domain_object.config.model.encoder)
        self.domain_object.set_encoder(encoder)

    # ---- decoder ----
    def build_decoder(self):
        from cdsvae.domain.model.kvae.network import FullConvFrameDecoder
        decoder = FullConvFrameDecoder(
            dim_a        = self.domain_object.config.model.encoder.dim_a,
            out_channels = self.domain_object.config.model.encoder.in_channels,
        )
        self.domain_object.set_decoder(decoder)

    def build_lgssm_matrix_params(self):
        from cdsvae.domain.model.kvae.lgssm import MatrixParams
        matrix_params = MatrixParams(
            dim_a       = self.domain_object.config.model.encoder.dim_a,
            **self.domain_object.config.model.matrix_params,
            num_mixture = self.domain_object.config.model.dynamics_parameter_network.num_mixture
        )
        self.domain_object.set_matrix_params(matrix_params)

    # --- LGSSM ---
    def build_kalman_filter(self):
        from cdsvae.domain.model.kvae.lgssm.kalmanfilter import InitialStateProbability, InitialDummyObservation
        from cdsvae.domain.model.kvae.lgssm.kalmanfilter import KalmanFilter, KFPredict, KFUpdate
        from cdsvae.domain.model.kvae.lgssm.dpn import DynamicsParameterNetworkManager, DynamicsParameterNetwork
        # ---
        init_probability = InitialStateProbability(
            dim_z     = self.domain_object.config.model.matrix_params.dim_z,
            **self.domain_object.config.model.initial_state_probability
        )
        init_dummy_observation = InitialDummyObservation(dim_y=self.dim_y)
        dpn = DynamicsParameterNetwork(
            dim_y = self.dim_y,
            dim_u = self.domain_object.config.model.matrix_params.dim_u,
            **self.domain_object.config.model.dynamics_parameter_network,
        )
        dpn_manager   = DynamicsParameterNetworkManager(dpn)
        kf_predict    = KFPredict(self.domain_object.matrix_params)
        kf_update     = KFUpdate( self.domain_object.matrix_params)
        kalman_filter = KalmanFilter(
            init_state_probability = init_probability,
            init_dummy_observation = init_dummy_observation,
            dpn_manager            = dpn_manager,
            kf_predict             = kf_predict,
            kf_update              = kf_update,
        )
        self.domain_object.set_kalman_filter(kalman_filter)


    def build_kalman_smoother(self):
        from cdsvae.domain.model.kvae.lgssm.kalmansmoother import KFSmoothing
        from cdsvae.domain.model.kvae.lgssm.kalmansmoother import KalmanSmoother
        kf_smoothing    = KFSmoothing()
        kalman_smoother = KalmanSmoother(kf_smoothing)
        self.domain_object.set_kalman_smoother(kalman_smoother)


    def build_object_state_decoder(self):
        from cdsvae.domain.model.kvae.network import TwoLinearObjectStateDecoder
        object_state_decoder = TwoLinearObjectStateDecoder(
            dim_z      = self.dim_z,
            dim_hidden = self.domain_object.config.model.object_state_decoder.dim_hidden,
            dim_c      = self.domain_object.config.model.object_state_decoder.dim_c,
            norm       = "BatchNorm1d",
        )
        self.domain_object.set_object_state_decoder(object_state_decoder)
        '''
        ここまでしかできてないので，lossの追加とか optimizerの設定とか，まだやらないと．．．
        '''


    def get_domain_object(self) -> KVAEDomainObject:
        return self.domain_object

