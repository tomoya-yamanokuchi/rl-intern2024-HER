from typing import List
import xml.etree.ElementTree as ET


class TextureTree:
    def __init__(self, root):
        self.subelement = ET.SubElement(root, 'texture')

    def add(self, id: int):
        self.subelement.set("name", "object_geom_vis_{}_tex".format(id))
        self.subelement.set("type", "2d")
        self.subelement.set("file", "./pattern/pattern_default_crop.png")
        self.subelement.set("rgb1", "1 1 1")
