# from .instance.simulation.valve import ValveSimulationEnvironment, ValveStateFactory
# from .instance.simulation.valve import ValveSimulationEnvironment, ValveStateFactory

# from .instance.simulation.pushing.PushingSimulationEnvironment import PushingSimulationEnvironment
# from .instance.simulation.pushing.PushingStateFactory import PushingStateFactory

# from .instance.simulation.rectangular_pushing import RectangularPushingSimulationEnvironment

from .instance.simulation.cube import CubeSimulationEnvironment, CubeStateFactory

'''
・環境を生成するクラスです
・新たな独自環境を作成して切り替えたいときには条件分岐を追加することで対応できます
'''

class EnvironmentFactory:
    @staticmethod
    def create(env_name: str):
        assert type(env_name) == str

        # if   env_name == "sim_valve"               : return (ValveSimulationEnvironment, ValveStateFactory)
        # if   env_name == "sim_pushing"             : return (PushingSimulationEnvironment, PushingStateFactory)
        # if   env_name == "sim_rectangular_pushing" : return (RectangularPushingSimulationEnvironment, PushingStateFactory)
        # import ipdb; ipdb.set_trace()
        if   env_name == "sim_cube"                : return (CubeSimulationEnvironment, CubeStateFactory)
        # elif env_name == "real"          : return DClawRealEnvironment
        raise NotImplementedError()

