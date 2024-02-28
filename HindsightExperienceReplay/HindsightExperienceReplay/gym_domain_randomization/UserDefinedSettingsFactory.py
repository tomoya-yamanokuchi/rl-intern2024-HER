
class UserDefinedSettingsFactory():
    @staticmethod
    def generate(env_name: str):

        if env_name == 'Pendulum':
            from .UserDefinedSettings import UserDefinedSettings
            return UserDefinedSettings()

        if env_name == 'RobelDClawCube':
            from .RobelDClawCubeUserDefinedSettings import RobelDClawCubeUserDefinedSettings
            return RobelDClawCubeUserDefinedSettings()

        raise NotImplementedError()
