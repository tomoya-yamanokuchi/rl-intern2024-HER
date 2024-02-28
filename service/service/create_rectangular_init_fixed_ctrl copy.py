import numpy as np



def create_active_ctrl(step_fixed: int) -> np.ndarray:
    x_traj = np.linspace(start=0.78, stop=0.70, num=step_fixed+1, endpoint=True)[1:]
    y_traj = np.linspace(start=0.49, stop=0.63, num=step_fixed+1, endpoint=True)[1:]
    # return np.diff(x_traj), np.diff(y_traj)
    return x_traj, y_traj


def create_rectangular_init_fixed_ctrl(
        init_task_space_position,
        step_fixed: int
    ):
    dim_u          = 6
    ctrl           = np.stack([np.array(init_task_space_position) for i in range(step_fixed)])
    x_traj, y_traj = create_active_ctrl(step_fixed)
    ctrl[:, 0]     = x_traj
    ctrl[:, 1]     = y_traj
    # ---
    # import ipdb; ipdb.set_trace()
    return ctrl


