import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define column names for the optitrack data
optitrack_columns = [
    'Frame', 'Time (Seconds)', 'Quaternion_X', 'Quaternion_Y', 'Quaternion_Z', 'Quaternion_W', 'X', 'Y', 'Z'
]

# Step 2: Load the CSV files, skipping the first 7 lines for optitrack_recording.csv
odometry_df = pd.read_csv('odometry.csv')
optitrack_df = pd.read_csv('optitrack_recording.csv', skiprows=7, names=optitrack_columns)

# Check the sample time (Assume it's in seconds)
sample_time_odometry = odometry_df['sec'].diff().mean()
sample_time_optitrack = optitrack_df['Time (Seconds)'].diff().mean()
print(f'Sample time for odometry: {sample_time_odometry} seconds')
print(f'Sample time for optitrack: {sample_time_optitrack} seconds')

# Make time relative
odometry_df["sec"] = odometry_df["sec"] - odometry_df["sec"].iloc[0]

# Shift odometry data to make the starting point (0, 0)
odometry_df['pos_x'] -= odometry_df['pos_x'].iloc[0]
odometry_df['pos_y'] -= odometry_df['pos_y'].iloc[0]

# Shift Optitrack data to make the starting point (0, 0)
optitrack_df['X'] -= optitrack_df['X'].iloc[0]
optitrack_df['Y'] -= optitrack_df['Y'].iloc[0]
optitrack_df['Z'] -= optitrack_df['Z'].iloc[0]

# Convert optitrack data from mm to meters
optitrack_df[['X', 'Y', 'Z']] = optitrack_df[['X', 'Y', 'Z']] / 1000.0

# Plotting the initial data
fig, axs = plt.subplots(2, 1)

# Optitrack Data - Initial
axs[0].plot(optitrack_df['X'], optitrack_df['Z'], label='Optitrack', linestyle='-', marker='x', markersize=5)
axs[0].scatter([optitrack_df['X'].iloc[0]], [optitrack_df['Z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[0].scatter([optitrack_df['X'].iloc[-1]], [optitrack_df['Z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('Optitrack - Initial Data')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Z Coordinate')
axs[0].legend()

# Odometry Data - Initial
axs[1].plot(odometry_df['pos_x'], odometry_df['pos_y'], label='Odometry', linestyle='-', marker='x', markersize=5)
axs[1].scatter([odometry_df['pos_x'].iloc[0]], [odometry_df['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[1].scatter([odometry_df['pos_x'].iloc[-1]], [odometry_df['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('Odometry - Initial Data')
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Y Coordinate')
axs[1].legend()

plt.tight_layout()

# Step 4: Transform coordinates from FUR (Forward, Up, Right) to FLU (Forward, Left, Up)
# - X (FUR) -> X (FLU): Remains the same.
# - Y (FUR) -> Z (FLU): The Z coordinate in FUR becomes the Y coordinate in FLU.
# - Z (FUR) -> -Y (FLU): The Y coordinate in FUR becomes the negative Z coordinate in FLU.

optitrack_df['FLU_x'] = optitrack_df['X']  # X remains the same
optitrack_df['FLU_y'] = optitrack_df['Z'] * -1  # Z becomes Y
optitrack_df['FLU_z'] = optitrack_df['Y']  # Y becomes -Z

# Step 5: Putting into the relative frame (relative to the first point)
relative_frame_origin_optitrack = optitrack_df[['FLU_x', 'FLU_y', 'FLU_z']].iloc[0]
optitrack_df.loc[:, 'FLU_x'] -= relative_frame_origin_optitrack['FLU_x']
optitrack_df.loc[:, 'FLU_y'] -= relative_frame_origin_optitrack['FLU_y']
optitrack_df.loc[:, 'FLU_z'] -= relative_frame_origin_optitrack['FLU_z']

# Set start points for Optitrack at (0, 0, 0)
optitrack_df.loc[0, 'FLU_x'] = 0
optitrack_df.loc[0, 'FLU_y'] = 0
optitrack_df.loc[0, 'FLU_z'] = 0

# Step 6: Convert the time columns to datetime format
odometry_df['Time'] = pd.to_datetime(odometry_df['sec'], unit='s')
initial_odometry_time = odometry_df['Time'].iloc[0]
optitrack_df['Time'] = pd.to_timedelta(optitrack_df['Time (Seconds)'], unit='s') + initial_odometry_time

# Remove duplicates from both dataframes to avoid reindexing issues
optitrack_df = optitrack_df.drop_duplicates(subset='Time')
odometry_df = odometry_df.drop_duplicates(subset='Time')

# Ensure there are no duplicate timestamps after merging
optitrack_resampled = optitrack_df.set_index('Time')
optitrack_resampled = optitrack_resampled[~optitrack_resampled.index.duplicated(keep='first')]

odometry_df = odometry_df.set_index('Time')
odometry_df = odometry_df[~odometry_df.index.duplicated(keep='first')]

# Step 7: Resampling the optitrack data to have the same timestamps as the odometry data
optitrack_resampled = optitrack_resampled.reindex(
    odometry_df.index, method='nearest', tolerance=pd.Timedelta('100ms')
).reset_index()

# Step 8: Combine the dataframes
combined_dataframes = pd.concat([odometry_df.reset_index(), optitrack_resampled.reset_index()], axis=1)

# Step 9: Drop rows where merge could not find a match within the tolerance
combined_dataframes = combined_dataframes.dropna()

# Function to plot specific columns from odometry data
def plot_odometry(ax, label):
    ax.plot(combined_dataframes['pos_x'], combined_dataframes['pos_y'], label=label, linestyle='-', marker='x', markersize=5)
    ax.scatter([combined_dataframes['pos_x'].iloc[0]], [combined_dataframes['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
    ax.scatter([combined_dataframes['pos_x'].iloc[-1]], [combined_dataframes['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')

# Function to plot specific columns from optitrack data
def plot_optitrack(ax, label):
    ax.plot(combined_dataframes['FLU_x'], combined_dataframes['FLU_y'], label=label, linestyle='-', marker='x', markersize=5)
    ax.scatter([combined_dataframes['FLU_x'].iloc[0]], [combined_dataframes['FLU_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
    ax.scatter([combined_dataframes['FLU_x'].iloc[-1]], [combined_dataframes['FLU_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')

# Step 10: Plot the data with specified adjustments
fig, axs = plt.subplots(2, 1)

# Optitrack Data - X vs Z
plot_optitrack(axs[0], 'Optitrack (X vs Y)')
axs[0].set_title('Top Down View - Optitrack')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')
axs[0].legend()

# Odometry Data - X vs Y
plot_odometry(axs[1], 'Odometry (X vs Y)')
axs[1].set_title('Top Down View - Odometry')
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Y Coordinate')
axs[1].legend()

plt.tight_layout()

fig, axs = plt.subplots(2, 1)

# Transformed Optitrack Data - X vs Y
axs[0].plot(combined_dataframes['FLU_x'], combined_dataframes['FLU_y'], label='Optitrack (Transformed X vs Y)', linestyle='-', marker='x', markersize=5)
axs[0].scatter([combined_dataframes['FLU_x'].iloc[0]], [combined_dataframes['FLU_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[0].scatter([combined_dataframes['FLU_x'].iloc[-1]], [combined_dataframes['FLU_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('FLU Transformation - Optitrack')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')
axs[0].legend()

# Transformed Odometry Data - X vs Y
axs[1].plot(combined_dataframes['pos_x'], combined_dataframes['pos_y'], label='Odometry (Transformed X vs Y)', linestyle='-', marker='x', markersize=5)
axs[1].scatter([combined_dataframes['pos_x'].iloc[0]], [combined_dataframes['pos_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
axs[1].scatter([combined_dataframes['pos_x'].iloc[-1]], [combined_dataframes['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('FLU Transformation - Odometry')
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Y Coordinate')
axs[1].legend()

plt.tight_layout()

# 3D Plotting for both datasets
fig = plt.figure()

# 3D plot for Optitrack Data
ax = fig.add_subplot(121, projection='3d')
ax.plot(combined_dataframes['FLU_x'], combined_dataframes['FLU_y'], combined_dataframes['FLU_z'], label='Optitrack', linestyle='-', marker='x', markersize=5)
ax.scatter([combined_dataframes['FLU_x'].iloc[0]], [combined_dataframes['FLU_y'].iloc[0]], [combined_dataframes['FLU_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax.scatter([combined_dataframes['FLU_x'].iloc[-1]], [combined_dataframes['FLU_y'].iloc[-1]], [combined_dataframes['FLU_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax.set_title('3D View - Optitrack')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()

# 3D plot for Odometry Data
ax = fig.add_subplot(122, projection='3d')
ax.plot(combined_dataframes['pos_x'], combined_dataframes['pos_y'], combined_dataframes['pos_z'], label='Odometry', linestyle='-', marker='x', markersize=5)
ax.scatter([combined_dataframes['pos_x'].iloc[0]], [combined_dataframes['pos_y'].iloc[0]], [combined_dataframes['pos_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax.scatter([combined_dataframes['pos_x'].iloc[-1]], [combined_dataframes['pos_y'].iloc[-1]], [combined_dataframes['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax.set_title('3D View - Odometry')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()

plt.tight_layout()
plt.show()
