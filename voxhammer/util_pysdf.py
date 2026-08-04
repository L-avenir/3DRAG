import numpy as np
import open3d as o3d
import trimesh
from pysdf import SDF

def load_trimesh(model_path):
    scene = trimesh.load(model_path, force='mesh', process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.scene.Scene):
        mesh = trimesh.Trimesh()
        for obj in scene.geometry.values():
            mesh = trimesh.util.concatenate([mesh, obj])
    else:
        raise ValueError(f'Unknown mesh type at {model_path}.')
    return mesh

def load_and_create_sdf(trimesh_mesh):
    sdf = SDF(trimesh_mesh.vertices, trimesh_mesh.faces)
    return sdf

def check_points_with_sdf(points, sdf):
    distances = sdf(points)
    inside_mask = distances > 0
    return (inside_mask, distances)

def sample_uniform_points(min_bounds=-0.5, max_bounds=0.5, num_points=4096):
    points = np.random.uniform(min_bounds, max_bounds, (num_points, 3))
    return points

def save_colored_point_cloud(points, inside_mask, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((len(points), 3))
    colors[inside_mask] = [0, 1, 0]
    colors[~inside_mask] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)
if __name__ == '__main__':
    model_path = 'path/to/model.glb'
    mesh = trimesh.load(model_path, process=False)
    sampled_points = sample_uniform_points(mesh, num_points=4096)
    sdf = load_and_create_sdf(mesh)
    (inside_mask, distances) = check_points_with_sdf(sampled_points, sdf)
    output_path = 'sampled_points.ply'
    save_colored_point_cloud(sampled_points, inside_mask, output_path)