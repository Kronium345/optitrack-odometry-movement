import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
data_path = Path("isaacVslam/take_2/")

gt_odometry = pd.read_csv(data_path / "gt_odometry.csv")
pr_odometry = pd.read_csv(data_path / "pr_odometry.csv")

# Apply the transformation to pr_odometry (FLU frame)
pr_odometry[["pos_y", "pos_z", "y", "z"]] = -1 * pr_odometry[["pos_y", "pos_z", "y", "z"]]

# Make time relative
gt_odometry["sec"] = gt_odometry["sec"] - gt_odometry["sec"].iloc[0]
pr_odometry["sec"] = pr_odometry["sec"] - pr_odometry["sec"].iloc[0]

# Shift data to make the starting point (0, 0, 0)
gt_odometry['pos_x'] -= gt_odometry['pos_x'].iloc[0]
gt_odometry['pos_y'] -= gt_odometry['pos_y'].iloc[0]
gt_odometry['pos_z'] -= gt_odometry['pos_z'].iloc[0]

pr_odometry['pos_x'] -= pr_odometry['pos_x'].iloc[0]
pr_odometry['pos_y'] -= pr_odometry['pos_y'].iloc[0]
pr_odometry['pos_z'] -= pr_odometry['pos_z'].iloc[0]

# Check the sample time (Assume it's in seconds)
sample_time_gt_odometry = gt_odometry['sec'].diff().mean()
sample_time_pr_odometry = pr_odometry['sec'].diff().mean()
print(f'Sample time for groundtruth: {sample_time_gt_odometry} seconds')
print(f'Sample time for prediction: {sample_time_pr_odometry} seconds')

# Plotting the initial data after making time relative
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Prediction Data - Initial
axs[0].plot(pr_odometry['pos_x'], pr_odometry['pos_z'], label='Prediction', linestyle='-', marker='o', markersize=5)
axs[0].scatter([pr_odometry['pos_x'].iloc[0]], [pr_odometry['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[0].scatter([pr_odometry['pos_x'].iloc[-1]], [pr_odometry['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('Prediction - Initial Data')
axs[0].set_xlabel('pos_x Coordinate')
axs[0].set_ylabel('pos_z Coordinate')
axs[0].legend()

# Groundtruth Data - Initial
axs[1].plot(gt_odometry['pos_x'], gt_odometry['pos_z'], label='Groundtruth', linestyle='--', marker='x', markersize=5)
axs[1].scatter([gt_odometry['pos_x'].iloc[0]], [gt_odometry['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[1].scatter([gt_odometry['pos_x'].iloc[-1]], [gt_odometry['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('Groundtruth - Initial Data')
axs[1].set_xlabel('pos_x Coordinate')
axs[1].set_ylabel('pos_z Coordinate')
axs[1].legend()

plt.tight_layout()

# Plotting the top-down view (X vs Y) for both datasets
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Prediction Data - Top Down View (Corrected with FLU Transformation)
axs[0].plot(pr_odometry['pos_x'], -pr_odometry['pos_y'], label='Prediction (FLU)', linestyle='-', marker='o', markersize=5)
axs[0].scatter([pr_odometry['pos_x'].iloc[0]], [-pr_odometry['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[0].scatter([pr_odometry['pos_x'].iloc[-1]], [-pr_odometry['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('Top Down View - Prediction')
axs[0].set_xlabel('pos_x Coordinate')
axs[0].set_ylabel('pos_y Coordinate')
axs[0].legend()

# Groundtruth Data - Top Down View
axs[1].plot(gt_odometry['pos_x'], -gt_odometry['pos_y'], label='Groundtruth', linestyle='--', marker='x', markersize=5)
axs[1].scatter([gt_odometry['pos_x'].iloc[0]], [-gt_odometry['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[1].scatter([gt_odometry['pos_x'].iloc[-1]], [-gt_odometry['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('Top Down View - Groundtruth')
axs[1].set_xlabel('pos_x Coordinate')
axs[1].set_ylabel('pos_y Coordinate')
axs[1].legend()

plt.tight_layout()

# Synchronize data based on movement using the Pythagorean theorem
def calculate_movement(df, x_col, y_col, z_col):
    return np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)

gt_odometry['movement'] = calculate_movement(gt_odometry, 'pos_x', 'pos_y', 'pos_z')
pr_odometry['movement'] = calculate_movement(pr_odometry, 'pos_x', 'pos_y', 'pos_z')

threshold = 0.5  # 10 cm
gt_odometry_sync = gt_odometry[gt_odometry['movement'] >= threshold].reset_index(drop=True)
pr_odometry_sync = pr_odometry[pr_odometry['movement'] >= threshold].reset_index(drop=True)

# Plotting the synchronized data
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Prediction Data - Synchronized with FLU
axs[0].plot(pr_odometry_sync['pos_x'], pr_odometry_sync['pos_y'], label='Prediction', linestyle='-', marker='o', markersize=5)
axs[0].scatter([pr_odometry_sync['pos_x'].iloc[0]], [pr_odometry_sync['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[0].scatter([pr_odometry_sync['pos_x'].iloc[-1]], [pr_odometry_sync['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('Prediction - FLU Transformation')
axs[0].set_xlabel('pos_x Coordinate')
axs[0].set_ylabel('pos_y Coordinate')
axs[0].legend()

# Groundtruth Data - Synchronized
axs[1].plot(gt_odometry_sync['pos_x'], gt_odometry_sync['pos_y'], label='Groundtruth', linestyle='--', marker='x', markersize=5)
axs[1].scatter([gt_odometry_sync['pos_x'].iloc[0]], [gt_odometry_sync['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[1].scatter([gt_odometry_sync['pos_x'].iloc[-1]], [gt_odometry_sync['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('Groundtruth - FLR Transformation')
axs[1].set_xlabel('pos_x Coordinate')
axs[1].set_ylabel('pos_y Coordinate')
axs[1].legend()

plt.tight_layout()

# 3D Plotting for both datasets
fig = plt.figure(figsize=(15, 7))

# 3D plot for Prediction Data with FLU
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(pr_odometry_sync['pos_x'], pr_odometry_sync['pos_y'], pr_odometry_sync['pos_z'], label='Prediction', linestyle='-', marker='o', markersize=5)
ax1.scatter([pr_odometry_sync['pos_x'].iloc[0]], [pr_odometry_sync['pos_y'].iloc[0]], [pr_odometry_sync['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax1.scatter([pr_odometry_sync['pos_x'].iloc[-1]], [pr_odometry_sync['pos_y'].iloc[-1]], [pr_odometry_sync['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax1.set_title('3D View - Prediction with FLU')
ax1.set_xlabel('pos_x Coordinate')
ax1.set_ylabel('pos_y Coordinate')
ax1.set_zlabel('pos_z Coordinate')

ax1.legend()

# 3D plot for Groundtruth Data with FLR
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(gt_odometry_sync['pos_x'], gt_odometry_sync['pos_y'], gt_odometry_sync['pos_z'], label='Groundtruth', linestyle='--', marker='x', markersize=5)
ax2.scatter([gt_odometry_sync['pos_x'].iloc[0]], [gt_odometry_sync['pos_y'].iloc[0]], [gt_odometry_sync['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax2.scatter([gt_odometry_sync['pos_x'].iloc[-1]], [gt_odometry_sync['pos_y'].iloc[-1]], [gt_odometry_sync['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax2.set_title('3D View - Groundtruth with FLR')
ax2.set_xlabel('pos_x Coordinate')
ax2.set_ylabel('pos_y Coordinate')
ax2.set_zlabel('pos_z Coordinate')
ax2.legend()

plt.tight_layout()
plt.show()

# Debugging: Ensure the red X is at the endpoint for all plots
print("Groundtruth synchronized data end point:", gt_odometry_sync[['pos_x', 'pos_y', 'pos_z']].iloc[-1])
print("Prediction synchronized data end point:", pr_odometry_sync[['pos_x', 'pos_y', 'pos_z']].iloc[-1])

