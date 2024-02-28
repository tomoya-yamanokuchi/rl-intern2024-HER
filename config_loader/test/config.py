from argparse import ArgumentParser
from typing import List, Dict, Any


class FileConfig:
    file   : str       = "./myfile.txt"
    count  : int       = 3
    numbers: List[int] = [0, 1, 2]
    flag   : bool      = True


class ModelConfig:
    name   : str       = "./myfile.txt"
    dim_a  : int       = 3
    dim_z  : List[int] = [0, 1, 2]
    epoch  : int       = 50


    def __init__(self, parser: ArgumentParser):
        self.parser = parser

    def list_annotated_attributes(self) -> Dict[str, Any]:
        annotated_attributes = {}
        for attr_name, attr_type in self.__annotations__.items():
            annotated_attributes[attr_name] = {'value': getattr(self, attr_name), 'type': attr_type}
        return annotated_attributes

    def validate_type(self, attr_name: str, value: Any, expected_type: type) -> bool:
        if not isinstance(value, expected_type):
            print(f"Type mismatch for {attr_name}: Expected {expected_type}, got {type(value)}")
            return False
        return True

    def parse(self) -> 'FileConfig':
        annotated_attributes = self.list_annotated_attributes()
        for attr_name, val in annotated_attributes.items():
            self.parser.add_argument('--{}'.format(attr_name), type=val["type"], help=attr_name)
        args = self.parser.parse_args()

        for attr_name, attr_info in annotated_attributes.items():
            arg_value = getattr(args, attr_name)
            if arg_value is not None:
                if self.validate_type(attr_name, arg_value, attr_info['type']):
                    setattr(self, attr_name, arg_value)

        return self

if __name__ == "__main__":
    parser = ArgumentParser()
    config = FileConfig(parser)
    parsed_config = config.parse()

    print(parsed_config.file)
    print(parsed_config.count)
    print(parsed_config.numbers)
    print(parsed_config.flag)
