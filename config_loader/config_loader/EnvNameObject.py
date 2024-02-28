

class EnvNameObject:
    def __init__(self, env_name: str):
        self.__env_name                  = env_name
        self.__env_name_without_instance = self.env_name.split("/")[-1]

    @property
    def env_name(self) -> str:
        return self.__env_name

    @property
    def env_name_without_instance(self) -> str:
        return self.__env_name_without_instance
