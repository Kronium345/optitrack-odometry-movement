import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data_path = Path("isaacVslam/take_1_new/take_1/")
gt_odometry = pd.read_csv(data_path / "gt_odometry.csv")
pr_odometry = pd.read_csv(data_path / "pr_odometry.csv")

# Align the time ranges
start_time = max(gt_odometry.iloc[0]["sec"], pr_odometry.iloc[0]["sec"])
end_time = min(gt_odometry.iloc[-1]["sec"], pr_odometry.iloc[-1]["sec"])
gt_odometry = gt_odometry[gt_odometry['sec'].between(start_time, end_time)]
pr_odometry = pr_odometry[pr_odometry['sec'].between(start_time, end_time)]

# Convert 'sec' to TimedeltaIndex and resample the data
gt_odometry['sec'] = pd.to_timedelta(gt_odometry['sec'] - gt_odometry['sec'].iloc[0], unit='s')
pr_odometry['sec'] = pd.to_timedelta(pr_odometry['sec'] - pr_odometry['sec'].iloc[0], unit='s')

# Resample the data based on time (100ms intervals)
gt_odometry = gt_odometry.set_index('sec').resample('100ms').first().interpolate().reset_index()
pr_odometry = pr_odometry.set_index('sec').resample('100ms').first().interpolate().reset_index()

# Apply the transformation to both datasets
def apply_transformation(row, T0_inv):
    r = R.from_quat([row['x'], row['y'], row['z'], row['w']])
    t = np.array([row['pos_x'], row['pos_y'], row['pos_z']])

    Tn = np.eye(4)
    Tn[:3, :3] = r.as_matrix()
    Tn[:3, 3] = t

    Tn_transformed = np.matmul(T0_inv, Tn)
    transformed_pos = Tn_transformed[:3, 3]
    transformed_rot = R.from_matrix(Tn_transformed[:3, :3])

    return pd.Series({
        'sec': row["sec"],  # Convert back to seconds for the final DataFrame
        'pos_x': transformed_pos[0],
        'pos_y': transformed_pos[1],
        'pos_z': transformed_pos[2],
        'x': transformed_rot.as_quat()[0],
        'y': transformed_rot.as_quat()[1],
        'z': transformed_rot.as_quat()[2],
        'w': transformed_rot.as_quat()[3]
    })

# Apply the transformation to the prediction data
first_row_pred = pr_odometry.iloc[0]
r_pred = R.from_quat([first_row_pred['x'], first_row_pred['y'], first_row_pred['z'], first_row_pred['w']])
t_pred = np.array([first_row_pred['pos_x'], first_row_pred['pos_y'], first_row_pred['pos_z']])

T0_pred = np.eye(4)
T0_pred[:3, :3] = r_pred.as_matrix()
T0_pred[:3, 3] = t_pred
T0_pred_inv = np.linalg.inv(T0_pred)

pr_odometry = pr_odometry.apply(apply_transformation, axis=1, T0_inv=T0_pred_inv)

# Apply the transformation to the ground truth data
first_row_gt = gt_odometry.iloc[0]
r_gt = R.from_quat([first_row_gt['x'], first_row_gt['y'], first_row_gt['z'], first_row_gt['w']])
t_gt = np.array([first_row_gt['pos_x'], first_row_gt['pos_y'], first_row_gt['pos_z']])

T0_gt = np.eye(4)
T0_gt[:3, :3] = r_gt.as_matrix()
T0_gt[:3, 3] = t_gt
T0_gt_inv = np.linalg.inv(T0_gt)

gt_odometry = gt_odometry.apply(apply_transformation, axis=1, T0_inv=T0_gt_inv)

# Calculate Euclidean distances between ground truth and prediction for all points
d_error = []
for gt_row, pr_row in zip(gt_odometry.iterrows(), pr_odometry.iterrows()):
    delta_x = gt_row[1]['pos_x'] - pr_row[1]['pos_x']
    delta_y = gt_row[1]['pos_y'] - pr_row[1]['pos_y']
    delta_z = gt_row[1]['pos_z'] - pr_row[1]['pos_z']
    distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    d_error.append(distance)

print(f"Average distance error: {np.mean(d_error):.2f} m")

# Plot the results
fig = plt.figure(figsize=(14, 7))

# 3D plot of the transformed ground truth and prediction data
ax = fig.add_subplot(121, projection='3d')
ax.plot(gt_odometry['pos_x'], gt_odometry['pos_y'], gt_odometry['pos_z'], label='Transformed Ground Truth', color='blue')
ax.plot(pr_odometry['pos_x'], pr_odometry['pos_y'], pr_odometry['pos_z'], label='Prediction', color='orange')
ax.set_title("3D Trajectory")
ax.set_xlabel('pos_x')
ax.set_ylabel('pos_y')
ax.set_zlabel('pos_z')
ax.legend()

# Plot the error over time
ax2 = fig.add_subplot(122)
ax2.plot(d_error)
ax2.set_title("Distance Error Over Time")
ax2.set_xlabel('Index')
ax2.set_ylabel('Error (m)')

plt.savefig("results_two_points_corrected_3d.png")
plt.show()

# Print the average distance
print(f"The average Euclidean distance between the transformed data is: {np.mean(d_error):.3f} units.")
