from argparse import ArgumentParser
from typing import List, Type, Any, Dict

class BaseConfig:
    def __init__(self, parser: ArgumentParser):
        self.parser = parser
        self.add_arguments()

    def add_arguments(self):
        for name, type_ in self.__annotations__.items():
            default_value = getattr(self, name)
            self.parser.add_argument(f'--{name}', type=type_, default=default_value)

    def update_from_args(self, args):
        for name in self.__annotations__.keys():
            if hasattr(args, name):
                setattr(self, name, getattr(args, name))

class FileConfig(BaseConfig):
    file: str = "./myfile.txt"
    count: int = 3
    numbers: List[int] = [0, 1, 2]
    flag: bool = True

class ModelConfig(BaseConfig):
    name: str = "MyModel"
    dim_a: int = 3
    dim_z: List[int] = [0, 1, 2]
    epoch: int = 50
    file_config: FileConfig = None

    def __init__(self, parser):
        self.file_config = FileConfig(parser)  # Initialize this first
        super().__init__(parser)  # Then call the parent constructor

    def update_from_args(self, args):
        super().update_from_args(args)
        self.file_config.update_from_args(args)

if __name__ == "__main__":
    parser = ArgumentParser()
    config = ModelConfig(parser)
    args = parser.parse_args()

    # Update config objects from argparse results
    config.update_from_args(args)

    print(args.name)
    print(args.dim_a)
    print(args.file)
    print(args.count)
    print("----------")

    print(config.name)
    print(config.dim_a)
    print(config.file_config.file)
    print(config.file_config.count)
