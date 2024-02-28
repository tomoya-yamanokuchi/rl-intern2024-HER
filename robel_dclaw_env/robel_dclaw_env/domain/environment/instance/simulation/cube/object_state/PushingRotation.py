import numpy as np
from dataclasses import dataclass
from robel_dclaw_env.custom_service import dimension_assetion


@dataclass(frozen=True)
class PushingRotation:
    value: np.ndarray

    def __post_init__(self):
        dimension_assetion(self.value, dim=3) # 無関係の次元含めて全次元使う古いやつ
