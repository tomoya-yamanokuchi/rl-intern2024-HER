from typing import List
import xml.etree.ElementTree as ET


class VisualGeometryTree:
    def __init__(self, root, type, size, pos):
        self.root = root
        self.geom = ET.SubElement(root, 'geom')
        self.geom.set("type", type)
        self.geom.set("size", size)
        self.geom.set("pos" , pos)

    # def add_with_rgba(self, id: int, rgba: str = "0 0 1 1"):
    #     self.geom.set("class", "pushing_object_viz")
    #     self.geom.set("name", "object_geom_vis_{}".format(id))
    #     self.geom.set("rgba", rgba)

    def add_with_material(self, id: int):
        self.geom.set("class", "pushing_object_viz")
        self.geom.set("name", "object_geom_vis_{}".format(id))
        self.geom.set("material", "object_geom_vis_{}_mat".format(id))

    # def add(self, id: int, rgba: str = "0 0 1 1"):
    #     self.geom.set("class", "pushing_object_viz")
    #     self.geom.set("name", "object_geom_vis_{}".format(id))
    #     self.geom.set("material", "object_geom_vis_{}_mat".format(id))
    #     self.geom.set("rgba", rgba)
