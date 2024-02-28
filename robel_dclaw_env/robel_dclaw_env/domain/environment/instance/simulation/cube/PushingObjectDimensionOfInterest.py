from transforms3d.euler import euler2quat, quat2euler
from ..base_environment import AbstractObjectStateDimensionOfInterest


class PushingObjectDimensionOfInterest(AbstractObjectStateDimensionOfInterest):
    def extract(self, qinfo):
        '''
        x = state.qpos or state.qvel
        '''
        return qinfo[18:(18+2)] # 最初のxy平面の位置だけ欲しいため+2にする
