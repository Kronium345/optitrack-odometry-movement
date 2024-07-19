import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define column names for the optitrack data
optitrack_columns = [
    'Frame', 'Time (Seconds)', 'X', 'Y', 'Z', 'Rotation_W', 'pos_x', 'pos_y', 'pos_z'
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

# Shift optitrack data to make the starting point (0, 0, 0)
optitrack_df['pos_x'] -= optitrack_df['pos_x'].iloc[0]
optitrack_df['pos_y'] -= optitrack_df['pos_y'].iloc[0]
optitrack_df['pos_z'] -= optitrack_df['pos_z'].iloc[0]

# Plotting the initial data
fig, axs = plt.subplots(2, 1)

# Optitrack Data - Initial
axs[0].plot(optitrack_df['pos_x'], optitrack_df['pos_z'], label='Optitrack', linestyle='-', marker='x', markersize=5)
axs[0].scatter([0], [0], color='green', marker='x', s=100, label='Start Point (0, 0)')
axs[0].scatter([optitrack_df['pos_x'].iloc[-1]], [optitrack_df['pos_z'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[0].set_title('(pos_x vs pos_z)')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Z Coordinate')
axs[0].legend()

# Odometry Data - Initial
axs[1].plot(odometry_df['pos_x'], odometry_df['pos_y'], label='Odometry', linestyle='-', marker='x', markersize=5)
axs[1].scatter([0], [0], color='green', marker='x', s=100, label='Start Point (0, 0)')
axs[1].scatter([odometry_df['pos_x'].iloc[-1]], [odometry_df['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')
axs[1].set_title('(pos_x vs pos_y)')
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Y Coordinate')
axs[1].legend()

plt.tight_layout()
plt.show()

# Step 4: Transform coordinates from FUR (Forward, Up, Right) to FLU (Forward, Left, Up)
optitrack_df['FLU_x'] = optitrack_df['X']
optitrack_df['FLU_y'] = optitrack_df['Y'] * -1  # Multiply Y by -1
optitrack_df['FLU_z'] = optitrack_df['Z']

optitrack_df['FLU_pos_x'] = optitrack_df['pos_x']
optitrack_df['FLU_pos_y'] = optitrack_df['pos_y'] * -1  # Multiply pos_y by -1
optitrack_df['FLU_pos_z'] = optitrack_df['pos_z']

# Step 5: Putting into the relative frame (relative to the first point)
relative_frame_origin = optitrack_df[['FLU_x', 'FLU_y', 'FLU_z']].iloc[0]
optitrack_df['FLU_x'] -= relative_frame_origin['FLU_x']
optitrack_df['FLU_y'] -= relative_frame_origin['FLU_y']
optitrack_df['FLU_z'] -= relative_frame_origin['FLU_z']

relative_frame_origin_prime = optitrack_df[['FLU_pos_x', 'FLU_pos_y', 'FLU_pos_z']].iloc[0]
optitrack_df['FLU_pos_x'] -= relative_frame_origin_prime['FLU_pos_x']
optitrack_df['FLU_pos_y'] -= relative_frame_origin_prime['FLU_pos_y']
optitrack_df['FLU_pos_z'] -= relative_frame_origin_prime['FLU_pos_z']

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
    ax.scatter([0], [0], color='green', marker='x', s=100, label='Start Point (0, 0)')
    ax.scatter([combined_dataframes['pos_x'].iloc[-1]], [combined_dataframes['pos_y'].iloc[-1]], color='red', marker='x', s=100, label='End Point')

# Function to plot specific columns from optitrack data
def plot_optitrack(ax, x_label, y_label):
    ax.plot(combined_dataframes[x_label], combined_dataframes[y_label], label=f'Optitrack ({x_label} vs {y_label})', linestyle='-', marker='x', markersize=5)
    ax.scatter([0], [0], color='green', marker='x', s=100, label='Start Point (0, 0)')
    ax.scatter([combined_dataframes[x_label].iloc[-1]], [combined_dataframes[y_label].iloc[-1]], color='red', marker='x', s=100, label='End Point')

# Step 10: Plot the data with specified adjustments
fig, axs = plt.subplots(2, 1, figsize=(15, 10))

# Optitrack Data - pos_x vs pos_y
plot_optitrack(axs[0], 'FLU_pos_x', 'FLU_pos_y')
axs[0].set_title('Top-Down View')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')
axs[0].legend()

# Odometry Data - pos_x vs pos_y
plot_odometry(axs[1], 'Odometry (pos_x vs pos_y)')
axs[1].set_title('Top-Down View')
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Y Coordinate')
axs[1].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(15, 10))

# Transformed Optitrack Data - pos_x vs pos_y
plot_optitrack(axs[0], 'FLU_pos_x', 'FLU_pos_y')
axs[0].set_title('FLU Transformed')
axs[0].set_xlabel('X Coordinate')
axs[0].set_ylabel('Y Coordinate')
axs[0].legend()

# Transformed Odometry Data - pos_x vs pos_y
plot_odometry(axs[1], 'Odometry (Transformed pos_x vs pos_y)')
axs[1].set_title('FLU Transformed')
axs[1].set_xlabel('X Coordinate')
axs[1].set_ylabel('Y Coordinate')
axs[1].legend()

plt.tight_layout()
plt.show()
