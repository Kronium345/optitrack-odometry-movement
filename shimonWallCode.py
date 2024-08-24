import open3d as o3d
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

class Point:
    def __init__(self, location):
        self.location = np.array(location)
        self.point_type = self.determine_point_type()
        self.apply_offset()

    def determine_point_type(self):
        y_value = self.location[1]
        if y_value < 50:
            return "bottom"
        elif 50 <= y_value < 1200:
            return "middle"
        else:
            return "top"

    def apply_offset(self):
        offsets = {
            "top": np.array([20, -12.5, -20]),
            "middle": np.array([0, -20, 0]),
            "bottom": np.array([0, -20, -12.5])
        }
        if self.point_type in offsets:
            self.location += offsets[self.point_type]

    def translate(self, vector):
        self.location += np.array(vector)


class Wall:
    def __init__(self, name, points):
        self.name = name
        self.points = points

    @classmethod
    def from_position_data(cls, name, position_data):
        points = []
        for i in range(0, position_data.shape[1], 3):
            point_location = position_data.iloc[:, i:i+3].mean().values
            point_location = Wall.convert_orientation(point_location, 'NUE', 'RUB') * 1000
            point = Point(location=point_location)
            points.append(point)
        if name == "wall_2":
            points.extend(cls.add_temporary_lower_points(points))
        return cls(name=name, points=points)

    @staticmethod
    def add_temporary_lower_points(existing_points):
        all_locations = np.array([p.location for p in existing_points])
        min_x, min_y, min_z = np.min(all_locations, axis=0)
        max_x, max_y, max_z = np.max(all_locations, axis=0)
        lower_points = [
            Point(location=np.array([min_x, min_y - 500, min_z])),
            Point(location=np.array([max_x, min_y - 500, min_z])),
        ]
        return lower_points

    @staticmethod
    def convert_orientation(vector, source, target):
        transformations = {
            'RUB': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'NUE': np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            'FLU': np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        }
        source_matrix = transformations[source]
        target_matrix = transformations[target]
        transform_matrix = np.linalg.inv(target_matrix) @ np.linalg.inv(source_matrix)
        return transform_matrix @ np.array(vector)

    def create_triangle_mesh(self):
        all_points = np.array([p.location for p in self.points])
        if len(all_points) < 3:
            return None
        return self.create_triangle_mesh_from_points(all_points)

    def visualize(self, vis):
        for point in self.points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
            sphere.translate(point.location)
            color = [1, 0, 0] if point.point_type == "top" else [0, 1, 0] if point.point_type == "middle" else [0, 0, 1]
            sphere.paint_uniform_color(color)
            vis.add_geometry(sphere)

        mesh = self.create_triangle_mesh()
        if mesh:
            vis.add_geometry(mesh)

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        vis.add_geometry(origin)

    @staticmethod
    def create_triangle_mesh_from_points(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        mesh, _ = pcd.compute_convex_hull()
        mesh.compute_vertex_normals()
        return mesh
    
    @staticmethod
    def add_intersection_points(vis, wall_3, wall_4):
        """Add intersection points where wall_3 and wall_4 meet."""
        # Extract the relevant bounds
        wall_3_z = np.mean([p.location[2] for p in wall_3.points])
        wall_4_x = np.mean([p.location[0] for p in wall_4.points])
        common_y = np.mean([p.location[1] for p in wall_3.points + wall_4.points])

        # Create intersection points
        upper_point = Point(location=[wall_4_x, common_y + 1000, wall_3_z])  # Adjust Y for upper
        lower_point = Point(location=[wall_4_x, common_y - 1000, wall_3_z])  # Adjust Y for lower

        wall_3.points.extend([upper_point, lower_point])
        wall_4.points.extend([upper_point, lower_point])

    @staticmethod
    def add_floor(vis, floor_height, width, depth):
        """Create a raised floor as a flat surface (box) in the visualization."""
        floor = o3d.geometry.TriangleMesh.create_box(width=width, height=20, depth=depth)
        floor.translate([-width/2, floor_height, -depth/2])
        floor.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray color for the floor
        vis.add_geometry(floor)

# Load the dataset
file_path = 'wall_data_2.csv'
wall_data = pd.read_csv(file_path, header=None, skiprows=3)

# Extract the relevant rows for identifying markers and positions
wall_names = wall_data.iloc[0]   # Wall names (row 4)
marker_ids = wall_data.iloc[1]   # Marker IDs (row 5)
data_type = wall_data.iloc[2]    # Data type (e.g., Rotation, Position) (row 6)

# Identify columns with markers and position data only
marker_columns = data_type[data_type == 'Position'].index

# Identify and process each wall dynamically
walls = {}
for wall_name in ['wall_1', 'wall_2', 'wall_3', 'wall_4']:
    wall_columns = wall_names[wall_names.str.contains(wall_name, na=False)].index.intersection(marker_columns)
    wall_positions = wall_data.iloc[4:, wall_columns].astype(float)
    walls[wall_name] = Wall.from_position_data(wall_name, wall_positions)


# Visualize the walls with triangle meshes
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add intersection points between wall_3 and wall_4
Wall.add_intersection_points(vis, walls['wall_3'], walls['wall_4'])

# Visualize each wall
for wall in walls.values():
    wall.visualize(vis)

# Example of how you might use the alignment vectors with the first wall
shimonsWall = walls['wall_1']
shimonsWall_pivot = shimonsWall.points[0].location
shimonsWall_end = shimonsWall.points[4].location
shimonsWall_vector = shimonsWall_end - shimonsWall_pivot
shimonsWall_vector = shimonsWall_vector / np.linalg.norm(shimonsWall_vector)
rotation_matrix = rotation_matrix_from_vectors([1, 0, 0], shimonsWall_vector)

# Example mesh alignment
for i in [0, 1, 3]: #pivot points for each of the panels
    wall_mesh = o3d.io.read_triangle_mesh("Panel_1256.stl")
    wall_mesh.compute_vertex_normals()
    wall_mesh.paint_uniform_color([0.5, 0.5, 0.5]) #grey
    wall_mesh.translate([-20, -2452.5, -20]) #offset
    wall_mesh.translate(shimonsWall.points[i].location) #location
    wall_mesh.rotate(R.from_matrix(rotation_matrix).as_matrix(), center=shimonsWall.points[i].location)
    vis.add_geometry(wall_mesh)

# Add the raised floor
floor_height = -100  # Adjust the height of the floor relative to your walls
floor_width = 10000  # Width of the floor
floor_depth = 10000  # Depth of the floor
Wall.add_floor(vis, floor_height=floor_height, width=floor_width, depth=floor_depth)

# Run visualization
vis.run()
vis.destroy_window()
