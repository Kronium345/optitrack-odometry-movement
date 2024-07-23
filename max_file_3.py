import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define column names for the optitrack data
optitrack_columns = [
    'Frame', 'Time (Seconds)', 'Quaternion_X', 'Quaternion_Y', 'Quaternion_Z', 'Quaternion_W', 'X', 'Y', 'Z'
]

# Load the CSV files, skipping the first 7 lines for optitrack_recording.csv
optitrack_df = pd.read_csv('optitrack_20240717_0.csv', skiprows=7, names=optitrack_columns)

# Debugging: Print the first few rows of the optitrack data
print("Optitrack Data (first few rows):")
print(optitrack_df.head())

# Check if 'Time (Seconds)' column is present in optitrack_df
if 'Time (Seconds)' not in optitrack_df.columns:
    raise KeyError("'Time (Seconds)' column not found in optitrack data")

# Check the sample time (Assume it's in seconds)
sample_time_optitrack = optitrack_df['Time (Seconds)'].diff().mean()
print(f'Sample time for optitrack: {sample_time_optitrack} seconds')

# Shift Optitrack data to make the starting point (0, 0, 0)
optitrack_df['X'] -= optitrack_df['X'].iloc[0]
optitrack_df['Y'] -= optitrack_df['Y'].iloc[0]
optitrack_df['Z'] -= optitrack_df['Z'].iloc[0]

# Convert optitrack data from mm to meters
optitrack_df[['X', 'Y', 'Z']] = optitrack_df[['X', 'Y', 'Z']] / 1000.0

# Plotting the initial data
# fig, ax = plt.subplots(1, 1)

# # Optitrack Data - Initial
# ax.plot(optitrack_df['X'], optitrack_df['Z'], label='Optitrack', linestyle='-', marker='x', markersize=5)
# ax.scatter([optitrack_df['X'].iloc[0]], [optitrack_df['Z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
# ax.scatter([optitrack_df['X'].iloc[-1]], [optitrack_df['Z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
# ax.set_title('Optitrack - Initial Data')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Z Coordinate')
# ax.legend()

# plt.tight_layout()
# plt.show()

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
initial_optitrack_time = pd.to_datetime(optitrack_df['Time (Seconds)'].iloc[0], unit='s')
optitrack_df['Time'] = pd.to_timedelta(optitrack_df['Time (Seconds)'], unit='s') + initial_optitrack_time

# Remove duplicates from the optitrack dataframe to avoid reindexing issues
optitrack_df = optitrack_df.drop_duplicates(subset='Time')

# Ensure there are no duplicate timestamps after merging
optitrack_resampled = optitrack_df.set_index('Time')
optitrack_resampled = optitrack_resampled[~optitrack_resampled.index.duplicated(keep='first')].reset_index()

# Function to plot specific columns from optitrack data
def plot_optitrack(ax, label):
    ax.plot(optitrack_resampled['FLU_x'], optitrack_resampled['FLU_y'], label=label, linestyle='-', marker='x', markersize=5)
    ax.scatter([optitrack_resampled['FLU_x'].iloc[0]], [optitrack_resampled['FLU_y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
    ax.scatter([optitrack_resampled['FLU_x'].iloc[-1]], [optitrack_resampled['FLU_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')

# Step 7: Plot the data with specified adjustments

# Optitrack Data - Initial X vs Z
fig, ax = plt.subplots(1, 1)
ax.plot(optitrack_df['X'], optitrack_df['Z'], label='Optitrack', linestyle='-', marker='x', markersize=5)
ax.scatter([optitrack_df['X'].iloc[0]], [optitrack_df['Z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax.scatter([optitrack_df['X'].iloc[-1]], [optitrack_df['Z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax.set_title('Optitrack - Initial Data')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Z Coordinate')
ax.legend()
plt.tight_layout()

# Optitrack Data - Top Down View (X vs Y)
fig, ax = plt.subplots(1, 1)
ax.plot(optitrack_df['X'], optitrack_df['Y'], label='Optitrack', linestyle='-', marker='x', markersize=5)
ax.scatter([optitrack_df['X'].iloc[0]], [optitrack_df['Y'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax.scatter([optitrack_df['X'].iloc[-1]], [optitrack_df['Y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax.set_title('Top Down View - Optitrack')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend()
plt.tight_layout()

# Optitrack Data - FLU X vs Y
fig, ax = plt.subplots(1, 1)
plot_optitrack(ax, 'Optitrack (Transformed X vs Y)')
ax.set_title('FLU Transformation - Optitrack')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend()
plt.tight_layout()

# 3D Plotting for optitrack data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(optitrack_resampled['FLU_x'], optitrack_resampled['FLU_y'], optitrack_resampled['FLU_z'], label='Optitrack', linestyle='-', marker='x', markersize=5)
ax.scatter([optitrack_resampled['FLU_x'].iloc[0]], [optitrack_resampled['FLU_y'].iloc[0]], [optitrack_resampled['FLU_z'].iloc[0]], color='green', marker='x', s=100, label='Start Point')
ax.scatter([optitrack_resampled['FLU_x'].iloc[-1]], [optitrack_resampled['FLU_y'].iloc[-1]], [optitrack_resampled['FLU_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
ax.set_title('3D View - Optitrack')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()
plt.tight_layout()
plt.show()
