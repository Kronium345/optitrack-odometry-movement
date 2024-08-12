import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Load your data
data_path = Path("isaacVslam/take_2/")
gt_odometry = pd.read_csv(data_path/ "gt_odometry.csv")
pr_odometry = pd.read_csv(data_path/ "pr_odometry.csv")

# Modify pr_odometry to be in the drone coordinate frame
pr_odometry[["pos_y", "pos_z", "y", "z"]] = -1 * pr_odometry[["pos_y", "pos_z", "y", "z"]]

start_time = max(gt_odometry.iloc[0]["sec"], pr_odometry.iloc[0]["sec"])
end_time = min(gt_odometry.iloc[-1]["sec"], pr_odometry.iloc[-1]["sec"])
gt_odometry = gt_odometry[gt_odometry['sec'].between(start_time, end_time)]
pr_odometry = pr_odometry[pr_odometry['sec'].between(start_time, end_time)]

# Create the transformation matrix  T0 from the first row
first_row = gt_odometry.iloc[0]
r = R.from_quat([first_row['x'], first_row['y'], first_row['z'], first_row['w']])
t = np.array([first_row['pos_x'], first_row['pos_y'], first_row['pos_z']])

T0 = np.eye(4)  # Create a 4x4 identity matrix to simplify the process of building the transformation matrix
T0[:3, :3] = r.as_matrix()  # Fill the rotation part
T0[:3, 3] = t  # Fill the translation part

# Inverse the transformation matrix (call this T0_inv - which represents T0^-1)
T0_inv = np.linalg.inv(T0)

# Apply T0_inv to every row in the pr_odometry
def apply_transformation(row):
    r_pred = R.from_quat([row['x'], row['y'], row['z'], row['w']])
    t_pred = np.array([row['pos_x'], row['pos_y'], row['pos_z']])

    Tn = np.eye(4)
    Tn[:3, :3] = r_pred.as_matrix()
    Tn[:3, 3] = t_pred

    # Transform the current prediction with the inverse of T0
    Tn_transformed = np.matmul(T0_inv, Tn)

    # Extract the transformed position and rotation
    transformed_pos = Tn_transformed[:3, 3]
    transformed_rot = R.from_matrix(Tn_transformed[:3, :3])

    return pd.Series({
        'pos_x': transformed_pos[0],
        'pos_y': transformed_pos[1],
        'pos_z': transformed_pos[2],
        'x': transformed_rot.as_quat()[0],
        'y': transformed_rot.as_quat()[1],
        'z': transformed_rot.as_quat()[2],
        'w': transformed_rot.as_quat()[3]
    })

# Applying the transformation to the entire pr_odometry DataFrame
pr_odometry_transformed = pr_odometry.apply(apply_transformation, axis=1)

# Plot the results
fig, axs = plt.subplots(1, 2)

gt_odometry.plot("pos_x", "pos_y", ax=axs[0])
axs[0].set_title("groundtruth")

pr_odometry_transformed.plot("pos_x", "pos_y", ax=axs[1])
axs[1].set_title("prediction transformed")
plt.savefig("results_transformed.png")
