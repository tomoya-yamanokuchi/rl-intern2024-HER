import sys
from copy import deepcopy
from pprint import pprint
import numpy as np
import mujoco_py
from .CtrlInterpolation import CtrlInterpolation
from ..base_environment.render import RenderingNullObjectManagerFactory
from ..base_environment.render import RenderingManagerParamDIct
from ..base_environment.viewer import ViewerNullManagerFactory


class BaseEnvironment:
    def __init__(self, config, use_render: bool):
        self.config             = config
        self.use_render         = use_render
        self.inplicit_step      = config.inplicit_step
        self.sim                = None
        self.viewer_manager     = None
        self.rendering_manager  = None
        self.renderImageDict    = None
        self.ctrl_interpolation = CtrlInterpolation(
            num_interpolation    = config.inplicit_step,
            endpoint_margin_step = config.ctrl_interpolation_margin_step,
        )

    def load_model_from_path(self, model_path):
        assert model_path is not None
        return mujoco_py.load_model_from_path(model_path)

    def set_sim(self, sim):
        self.sim = sim

    def build_viewer(self):
        assert self.sim is not None
        # if self.viewer_manager is None:
            # self.viewer_manager = ViewerManager(self.sim, self.config)
        self.viewer_manager = ViewerNullManagerFactory.create(self.use_render, self.sim, self.config)


    def view(self):
        # if self.use_render is None: return
        # assert self.viewer_manager is not None
        self.viewer_manager.view(self.renderImageDict)

    def build_render(self, canonical_rgb):
        assert self.sim is not None
        # if self.rendering_manager is None:
        paramDict = RenderingManagerParamDIct(
            sim            = self.sim,
            canonical_rgb  = canonical_rgb,
            config_render  = self.config.render,
            config_texture = self.config.texture,
            config_camera  = self.config.camera,
            config_light   = self.config.light,
        )
        self.rendering_manager = RenderingNullObjectManagerFactory.create(self.use_render, paramDict)


    def render(self):
        # assert self.viewer_manager is not None # to be initialized before rendering
        assert self.rendering_manager is not None
        self.renderImageDict = self.rendering_manager.render()
        return self.renderImageDict

    def get_state(self):
        return deepcopy(self.sim.data.get_state())

    def set_ctrl(self, joint_space_positioin):
        # shape_comparison = joint_space_positioin.shape == (9,)
        # print("Comparison result:", shape_comparison) # True or Falseが出力されるはず
        # -----
        assert joint_space_positioin.shape == (9,), print(" +++ joint_space_positioin.shape = ", joint_space_positioin.shape, " type: ", type(joint_space_positioin.shape))
        self.sim.data.ctrl[:9] = joint_space_positioin

    @property
    def ctrl(self):
        return deepcopy(self.sim.data.ctrl)

    def step(self, is_view=False):
        self.__step_with_inplicit_step(is_view)

    def __step_with_inplicit_step(self, is_view):
        '''
        ・一回の sim.step() では，制御入力で与えた目標位置まで到達しない
        ・そして sim.step() 当たりの移動量は mujoco 内部の物理モデルから計算されておりsimに固有である
        ・この固有移動量は実機環境におけるロボットの移動量とは異なるため, これらのsimとrealの間にはdynamics reality gapが存在する
        ・モデルベースの Sim-to-Real では特に, 1ステップの状態遷移の違いがそのままdynamics reality-gapとなり
          性能低下を引き起こす恐れがあるため，なるべくこれを避けたい
        ・なので，これを避けるために複数回の sim.step() を内包して目標状態まで到達することを保証できるような
          当該関数を作成して用いる
        ・その際に，現在のロボットの関節角度と与えた目標関節角度が大きく離れていると, 速度・加速度の大きな動きになり，
          操作物体をはじくような動作がでてきてしまうため，中間目標角度を補完しながら生成して実行できるようにしている
        '''
        interpolated_ctrl = self.ctrl_interpolation.interpolate(
            current_joint_position = self.sim.get_state().qpos[:9],
            target_joint_position  = self.ctrl[:9],
        )
        # print("self.ctrl[:9], = ", self.ctrl[:9])
        # import ipdb; ipdb.set_trace()
        for i in range(self.inplicit_step):
            self.set_ctrl(interpolated_ctrl[i])
            self.sim.step()
            if is_view:
                if self.config.viewer.is_Offscreen : continue
                else                               : self.view()



