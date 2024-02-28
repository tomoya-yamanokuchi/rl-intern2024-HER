from typing import List
import xml.etree.ElementTree as ET
from .VisualGeometryTree import VisualGeometryTree
from .PhysicalGeometryTree import PhysicalGeometryTree
from .TextureTree import TextureTree
from .MaterialTree import MaterialTree


class MultiRectangularBodyTree:
    def __init__(self, type: str):
        '''
            type: [sphere, capsule, ellipsoid, cylinder, box, .... etc]
            rgba: ex) "0 0 1 1"
        '''
        self.root = ET.Element('body')
        self.root.set( "name", "pushing_object")
        # self.root.set(  "pos", "0 0 .01001")
        self.root.set(  "pos", "0 0 0")
        self.root.set("euler", "0 0 0")
        # common parameter
        self.type = type

    def add_geometry(self, pos, size, mass, friction, id: int):
        VisualGeometryTree(self.root, self.type, size, pos).add_with_material(id)
        # VisualGeometryTree(self.root, self.type, size, pos).add(id, rgba)
        PhysicalGeometryTree(self.root, self.type, size, pos).add(mass, friction)
