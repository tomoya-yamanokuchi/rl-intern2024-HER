from domain_object_builder import DomainObjectBuilder
from domain_object_director import GenerateNewXMLModelObjectDirector
from usecase_object import GenerateNewXMLModel


class Usecase:
    @staticmethod
    def run(domain_object):
        random_dc = GenerateNewXMLModel(domain_object)
        random_dc.time_start()
        random_dc.generate_xml_model()
        random_dc.time_stop()


if __name__ == "__main__":
    builder       = DomainObjectBuilder()
    director      = GenerateNewXMLModelObjectDirector()
    domain_object = director.construct(builder, env_name="sim/pushing")
    Usecase.run(domain_object)


