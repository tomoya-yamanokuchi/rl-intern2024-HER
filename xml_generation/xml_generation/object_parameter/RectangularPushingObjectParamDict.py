from typing import TypedDict


class RectangularPushingObjectParamDict(TypedDict):
    x_half_size        : float
    y_half_size        : float
    z_half_size        : float
    # ---
    mass               : float
    # ---
    sliding_friction   : float
    torsional_friction : float
    rolling_friction   : float
