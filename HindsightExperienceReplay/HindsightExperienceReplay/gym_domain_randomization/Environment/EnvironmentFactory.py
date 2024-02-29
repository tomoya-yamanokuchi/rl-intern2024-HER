
class EnvironmentFactory():
    def __init__(self, userDefinedSettings):
        self.ENVIRONMENT_NAME = userDefinedSettings.ENVIRONMENT_NAME
        self.userDefinedSettings = userDefinedSettings

    def generate(self, domain_num=None):

        if self.ENVIRONMENT_NAME == 'Pendulum':
            from .Pendulum.Pendulum import Pendulum
            return Pendulum(self.userDefinedSettings)

        if self.ENVIRONMENT_NAME == 'RobelDClawCube':
            from .RobelDClawCube.RobelDClawCube import RobelDClawCube
            from domain_object_builder import DomainObjectBuilder
            from domain_object_director import CheckEnvRolloutDirector
            # ---
            builder       = DomainObjectBuilder()
            director      = CheckEnvRolloutDirector()
            domain_object = director.construct(builder, env_name="sim/cube")

            return RobelDClawCube(domain_object, self.userDefinedSettings)
