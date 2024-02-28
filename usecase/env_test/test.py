import numpy as np




x_traj = np.linspace(
    start    = 0.75,
    stop     = 0.6,
    num      = 7,
    endpoint = True,
)

print("diff(x_traj) = ", np.diff(x_traj))
print("x_traj = ", x_traj)

# ------------

y_traj = np.linspace(
    start    = 0.5,
    stop     = 0.55,
    num      = 7,
    endpoint = True,
)

print("diff(y_traj) = ", np.diff(y_traj))
print("y_traj = ", y_traj)
