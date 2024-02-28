import numpy as np
from torch_numpy_converter import to_tensor


def create_active_ctrl(step_fixed: int) -> np.ndarray:
    x_traj = np.linspace(start=0.78, stop=0.70, num=step_fixed+1, endpoint=True)
    y_traj = np.linspace(start=0.49, stop=0.63, num=step_fixed+1, endpoint=True)
    return np.diff(x_traj), np.diff(y_traj)


def create_rectangular_init_fixed_ctrl(step_fixed: int):
    dim_u          = 6
    # ctrl           = np.zeros([step_fixed, dim_u])
    x_diff, y_diff = create_active_ctrl(step_fixed)
    # ctrl[:, 0]     = x_diff
    # ctrl[:, 1]     = y_diff
    # import ipdb; ipdb.set_trace()
    task_space_diff_ctrl = np.stack([x_diff, y_diff], axis=-1)
    # ---
    # import ipdb; ipdb.set_trace()
    return to_tensor(task_space_diff_ctrl)


