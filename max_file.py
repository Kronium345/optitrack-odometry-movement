import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Step 1: Define column names for the optitrack data
optitrack_columns = [
    'Frame', 'Time (Seconds)', 'Quaternion_X', 'Quaternion_Y', 'Quaternion_Z', 'Quaternion_W', 'pos_x', 'pos_y', 'pos_z'
]

# Step 2: Load the CSV files, skipping the first 7 lines for optitrack_recording.csv
odometry_df = pd.read_csv('odometry.csv')
optitrack_df = pd.read_csv('optitrack_recording.csv', skiprows=7, names=optitrack_columns)

# Make time relative
odometry_df["sec"] = odometry_df["sec"] - odometry_df["sec"].iloc[0]
optitrack_df["Time (Seconds)"] = optitrack_df["Time (Seconds)"] - optitrack_df["Time (Seconds)"].iloc[0]

# Shift data to make the starting point (0, 0, 0)
odometry_df['pos_x'] -= odometry_df['pos_x'].iloc[0]
odometry_df['pos_y'] -= odometry_df['pos_y'].iloc[0]
odometry_df['pos_z'] -= odometry_df['pos_z'].iloc[0]

optitrack_df['pos_x'] -= optitrack_df['pos_x'].iloc[0]
optitrack_df['pos_y'] -= optitrack_df['pos_y'].iloc[0]
optitrack_df['pos_z'] -= optitrack_df['pos_z'].iloc[0]

# Check the sample time (Assume it's in seconds)
sample_time_odometry = odometry_df['sec'].diff().mean()
sample_time_optitrack = optitrack_df['Time (Seconds)'].diff().mean()
print(f'Sample time for odometry: {sample_time_odometry} seconds')
print(f'Sample time for optitrack: {sample_time_optitrack} seconds')

# Plotting the initial data after making time relative
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Optitrack Data - Initial
axs[0].plot(optitrack_df['pos_x'], optitrack_df['pos_z'], label='Optitrack', linestyle='-', marker='o', markersize=5)
axs[0].scatter([optitrack_df['pos_x'].iloc[0]], [optitrack_df['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[0].scatter([optitrack_df['pos_x'].iloc[-1]], [optitrack_df['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('Optitrack - Initial Data')
axs[0].set_xlabel('pos_x Coordinate')
axs[0].set_ylabel('pos_z Coordinate')
axs[0].legend()

# Odometry Data - Initial
axs[1].plot(odometry_df['pos_x'], odometry_df['pos_z'], label='Odometry', linestyle='--', marker='x', markersize=5)
axs[1].scatter([odometry_df['pos_x'].iloc[0]], [odometry_df['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[1].scatter([odometry_df['pos_x'].iloc[-1]], [odometry_df['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('Odometry - Initial Data')
axs[1].set_xlabel('pos_x Coordinate')
axs[1].set_ylabel('pos_z Coordinate')
axs[1].legend()

plt.tight_layout()

# Plotting the top-down view (X vs Y) for both datasets
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Optitrack Data - Top Down View
axs[0].plot(optitrack_df['pos_x'], optitrack_df['pos_z'], label='Optitrack', linestyle='-', marker='o', markersize=5)
axs[0].scatter([optitrack_df['pos_x'].iloc[0]], [optitrack_df['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[0].scatter([optitrack_df['pos_x'].iloc[-1]], [optitrack_df['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('Top Down View - Optitrack')
axs[0].set_xlabel('pos_x Coordinate')
axs[0].set_ylabel('pos_z Coordinate')
axs[0].legend()

# Odometry Data - Top Down View
axs[1].plot(odometry_df['pos_x'], -odometry_df['pos_y'], label='Odometry', linestyle='--', marker='x', markersize=5)
axs[1].scatter([odometry_df['pos_x'].iloc[0]], [-odometry_df['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[1].scatter([odometry_df['pos_x'].iloc[-1]], [-odometry_df['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('Top Down View - Odometry')
axs[1].set_xlabel('pos_x Coordinate')
axs[1].set_ylabel('pos_y Coordinate')
axs[1].legend()


plt.tight_layout()

# Synchronize data based on movement using the Pythagorean theorem
def calculate_movement(df, x_col, y_col, z_col):
    return np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)

odometry_df['movement'] = calculate_movement(odometry_df, 'pos_x', 'pos_y', 'pos_z')
optitrack_df['movement'] = calculate_movement(optitrack_df, 'pos_x', 'pos_y', 'pos_z')

threshold = 0.5  # 10 cm
odometry_sync = odometry_df[odometry_df['movement'] >= threshold].reset_index(drop=True)
optitrack_sync = optitrack_df[optitrack_df['movement'] >= threshold].reset_index(drop=True)

# FLU Transformation for Synchronized Data (Optitrack only)
optitrack_sync['pos_x'], optitrack_sync['pos_y'], optitrack_sync['pos_z'] = (
    optitrack_sync['pos_x'], 
    -optitrack_sync['pos_z'], 
    optitrack_sync['pos_y']
)


# Plotting the synchronized data
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Optitrack Data - Synchronized with FLU
axs[0].plot(optitrack_sync['pos_x'], optitrack_sync['pos_y'], label='Optitrack', linestyle='-', marker='o', markersize=5)
axs[0].scatter([optitrack_sync['pos_x'].iloc[0]], [optitrack_sync['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[0].scatter([optitrack_sync['pos_x'].iloc[-1]], [optitrack_sync['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('Optitrack - FLU Transformation')
axs[0].set_xlabel('pos_x Coordinate')
axs[0].set_ylabel('pos_y Coordinate')
axs[0].legend()

# Odometry Data - Synchronized with FLU
axs[1].plot(odometry_sync['pos_x'], odometry_sync['pos_y'], label='Odometry', linestyle='--', marker='x', markersize=5)
axs[1].scatter([odometry_sync['pos_x'].iloc[0]], [odometry_sync['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[1].scatter([odometry_sync['pos_x'].iloc[-1]], [odometry_sync['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('Odometry - Synchronized with FLU Transformation')
axs[1].set_xlabel('pos_x Coordinate')
axs[1].set_ylabel('pos_y Coordinate')
axs[1].legend()

plt.tight_layout()

# 3D Plotting for both datasets
fig = plt.figure(figsize=(15, 7))

# 3D plot for Optitrack Data with FLU
ax = fig.add_subplot(121, projection='3d')
ax.plot(optitrack_sync['pos_x'], optitrack_sync['pos_y'], optitrack_sync['pos_z'], label='Optitrack', linestyle='-', marker='o', markersize=5)
ax.scatter([optitrack_sync['pos_x'].iloc[0]], [optitrack_sync['pos_y'].iloc[0]], [optitrack_sync['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax.scatter([optitrack_sync['pos_x'].iloc[-1]], [optitrack_sync['pos_y'].iloc[-1]], [optitrack_sync['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax.set_title('3D View - Optitrack with FLU')
ax.set_xlabel('pos_x Coordinate')
ax.set_ylabel('pos_y Coordinate')
ax.set_zlabel('pos_z Coordinate')
ax.legend()

# 3D plot for Odometry Data with FLU
ax = fig.add_subplot(122, projection='3d')
ax.plot(odometry_sync['pos_x'], odometry_sync['pos_y'], odometry_sync['pos_z'], label='Odometry', linestyle='--', marker='x', markersize=5)
ax.scatter([odometry_sync['pos_x'].iloc[0]], [odometry_sync['pos_y'].iloc[0]], [odometry_sync['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax.scatter([odometry_sync['pos_x'].iloc[-1]], [odometry_sync['pos_y'].iloc[-1]], [odometry_sync['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax.set_title('3D View - Odometry with FLU')
ax.set_xlabel('pos_x Coordinate')
ax.set_ylabel('pos_y Coordinate')
ax.set_zlabel('pos_z Coordinate')
ax.legend()

plt.tight_layout()
plt.show()

# Debugging: Ensure the red X is at the endpoint for all plots
print("Odometry synchronized data end point:", odometry_sync[['pos_x', 'pos_y', 'pos_z']].iloc[-1])
print("Optitrack synchronized data end point:", optitrack_sync[['pos_x', 'pos_y', 'pos_z']].iloc[-1])
