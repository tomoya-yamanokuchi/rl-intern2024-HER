from typing import TypedDict


class CylinderPushingObjectParamDict(TypedDict):
    radius             : float
    half_length        : float
    # ---
    mass               : float
    # ---
    sliding_friction   : float
    torsional_friction : float
    rolling_friction   : float
