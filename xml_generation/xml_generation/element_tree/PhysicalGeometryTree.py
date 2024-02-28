from typing import List
import xml.etree.ElementTree as ET


class PhysicalGeometryTree:
    def __init__(self, root, type, size, pos):
        self.geom = ET.SubElement(root, 'geom')
        self.geom.set("type", type)
        self.geom.set("size", size)
        self.geom.set("pos" , pos)
        self.rgba = "0 0 1 1"

    def add(self,
            mass     : float,
            friction : List[float],
        ):
        self.geom.set("class", "pushing_object_phy")
        self.geom.set("mass", str(mass))
        self.geom.set("friction", " ".join([str(fi) for fi in friction]))

