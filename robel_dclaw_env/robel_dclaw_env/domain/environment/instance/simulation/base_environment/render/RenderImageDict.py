from typing import TypedDict
from .rendering import ImageObject


class RenderImageDict(TypedDict):
    canonical     : ImageObject
    random_nonfix : ImageObject
