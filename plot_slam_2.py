import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import quaternion
from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy.interpolate import CubicSpline
# from scipy import interpolate
import warnings

from pathlib import Path


data_path = Path("isaacVslam/take_1_new/take_1/")

gt_odometry = pd.read_csv(data_path/ "gt_odometry.csv")
pr_odometry = pd.read_csv(data_path/ "pr_odometry.csv")

# Call R from Q, then load from 1st row

## modify pr to be in the drone coordinate frame 
pr_odometry[["pos_y", "pos_z", "y", "z"]] = -1 * pr_odometry[["pos_y", "pos_z", "y", "z"]]

start_time = max(gt_odometry.iloc[0]["sec"],pr_odometry.iloc[0]["sec"])
end_time = min(gt_odometry.iloc[-1]["sec"],pr_odometry.iloc[-1]["sec"])

gt_odometry = gt_odometry[gt_odometry['sec'].between(start_time, end_time)]
pr_odometry = pr_odometry[pr_odometry['sec'].between(start_time, end_time)]

print(gt_odometry.head())
print(pr_odometry.head())

fig, axs = plt.subplots(1,2)

gt_odometry.plot("pos_x", "pos_y", ax=axs[0])
axs[0].set_title("groundtruth")

pr_odometry.plot("pos_x", "pos_y", ax=axs[1])
axs[1].set_title("prediction")
plt.savefig("results.png")