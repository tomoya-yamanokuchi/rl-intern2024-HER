from typing import List
import xml.etree.ElementTree as ET


class MaterialTree:
    def __init__(self, root):
        self.subelement = ET.SubElement(root, 'material')

    def add(self, id: int):
        self.subelement.set("name", "object_geom_vis_{}_mat".format(id))
        self.subelement.set("shininess", "0.03")
        self.subelement.set("specular", "0.75")
        self.subelement.set("texture", "object_geom_vis_{}_tex".format(id))

