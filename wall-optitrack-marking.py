import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from scipy.spatial import ConvexHull

# Load the wall data
data_path = Path("isaacVslam/")
wall_data = pd.read_csv(data_path / "shimonsWall.csv", skiprows=7)

# Extract only the position data for the 7 points (assuming these are the correct columns)
# We will manually specify the columns that correspond to the positions
position_columns = [
    '1.893583', '2.390509', '-3.494969',  # Marker 1
    '1.908219', '-0.025477', '-1.080556',  # Marker 2
    '1.903388', '-0.033216', '-2.304659',  # Marker 3
    '1.903757', '-0.030828', '-2.264008',  # Marker 4
    '1.922021', '-0.039375', '-3.484170',  # Marker 5
    '1.918561', '2.406876', '-1.094792',  # Marker 6
    '1.900847', '2.399695', '-2.274045'   # Marker 7
]

# Reshape the position columns to get X, Y, Z for each marker
wall_points = wall_data[position_columns].to_numpy().reshape(-1, 3)

# Create a point cloud for the wall points
wall_pcd = o3d.geometry.PointCloud()
wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
wall_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for the wall points

# Fit a plane to the points to represent the wall
# Using SVD to find the best fitting plane
mean = np.mean(wall_points, axis=0)
centered_points = wall_points - mean
u, s, vh = np.linalg.svd(centered_points)
normal = vh[2, :]

# Now, create a grid on the plane for better visualization
grid_size = 10  # Adjust grid size for better visualization
x_grid, y_grid = np.meshgrid(
    np.linspace(np.min(wall_points[:, 0]), np.max(wall_points[:, 0]), grid_size),
    np.linspace(np.min(wall_points[:, 1]), np.max(wall_points[:, 1]), grid_size)
)

# Calculate corresponding z values on the plane
z_grid = (-normal[0] * (x_grid - mean[0]) - normal[1] * (y_grid - mean[1])) / normal[2] + mean[2]

# Convert the grid to points
plane_points = np.c_[x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]

# Create a mesh plane for visualization
hull = ConvexHull(plane_points[:, :2])
vertices = plane_points[hull.vertices]
triangles = [[0, i, i + 1] for i in range(1, len(vertices) - 1)]
triangles = np.array(triangles)

plane_mesh = o3d.geometry.TriangleMesh()
plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
plane_mesh.compute_vertex_normals()
plane_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray for the wall

# Visualize the point cloud and the wall plane together
o3d.visualization.draw_geometries([wall_pcd, plane_mesh])
