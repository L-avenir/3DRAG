import numpy as np
import open3d as o3d
from voxhammer.util_pysdf import *

def check_voxels_with_volume_intersection(source_points, sdf, voxel_size=1 / 64, threshold=0.1):
    inside_mask = np.zeros(len(source_points), dtype=bool)
    overlap_ratios = np.zeros(len(source_points))
    half_voxel = voxel_size / 2
    for (i, center) in enumerate(source_points):
        num_samples = 27
        samples = []
        for x in np.linspace(-half_voxel, half_voxel, 3):
            for y in np.linspace(-half_voxel, half_voxel, 3):
                for z in np.linspace(-half_voxel, half_voxel, 3):
                    sample_point = center + np.array([x, y, z])
                    samples.append(sample_point)
        samples = np.array(samples)
        distances = sdf(samples)
        inside_samples = distances > 0
        overlap_ratio = np.mean(inside_samples)
        overlap_ratios[i] = overlap_ratio
        if overlap_ratio >= threshold:
            inside_mask[i] = True
    return (inside_mask, overlap_ratios)

def check_voxels_with_distance_threshold(source_points, sdf, voxel_size=1 / 64, distance_threshold=0.05):
    distances = sdf(source_points)
    outside_mask = distances <= 0
    near_surface_mask = np.abs(distances) <= distance_threshold
    inside_mask = (distances > 0) | near_surface_mask & outside_mask
    return (inside_mask, distances)

def check_voxels_with_corner_sampling(source_points, sdf, voxel_size=1 / 64):
    inside_mask = np.zeros(len(source_points), dtype=bool)
    corner_counts = np.zeros(len(source_points), dtype=int)
    half_voxel = voxel_size / 2
    for (i, center) in enumerate(source_points):
        corners = []
        for x in [-half_voxel, half_voxel]:
            for y in [-half_voxel, half_voxel]:
                for z in [-half_voxel, half_voxel]:
                    corner = center + np.array([x, y, z])
                    corners.append(corner)
        corners = np.array(corners)
        distances = sdf(corners)
        inside_corners = distances > 0
        corner_count = np.sum(inside_corners)
        corner_counts[i] = corner_count
        if corner_count > 0:
            inside_mask[i] = True
    return (inside_mask, corner_counts)

def adaptive_voxel_filtering(source_points, sdf, voxel_size=1 / 64, method='volume'):
    if method == 'volume':
        if voxel_size >= 1 / 32:
            threshold = 0.05
        elif voxel_size >= 1 / 64:
            threshold = 0.1
        else:
            threshold = 0.2
        return check_voxels_with_volume_intersection(source_points, sdf, voxel_size, threshold)
    elif method == 'distance':
        distance_threshold = voxel_size * 0.5
        return check_voxels_with_distance_threshold(source_points, sdf, voxel_size, distance_threshold)
    elif method == 'corner':
        return check_voxels_with_corner_sampling(source_points, sdf, voxel_size)
    else:
        raise ValueError(f'Unknown method: {method}')

def process_voxels_with_improved_filtering(source_voxel_path, mask_model_path, output_path, method='volume', voxel_size=1 / 64, inside=False):
    mask_mesh = load_trimesh(mask_model_path)
    sdf = load_and_create_sdf(mask_mesh)
    source_pcd = o3d.io.read_point_cloud(source_voxel_path)
    source_points = np.asarray(source_pcd.points)
    print(source_points.shape)
    print(f'=== Filtering logic debug ===')
    (inside_mask, additional_info) = adaptive_voxel_filtering(source_points, sdf, voxel_size, method)
    mask = inside_mask if inside else ~inside_mask
    print(f'inside parameter value: {inside}')
    print(f'Number of voxels intersecting with the mask: {np.sum(inside_mask)}')
    print(f'Number of voxels not intersecting with the mask: {len(source_points) - np.sum(inside_mask)}')
    print(f'Final retained mask (True) count: {np.sum(mask)}')
    target_voxel_points = source_points[mask]
    print(target_voxel_points.shape)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_voxel_points)
    o3d.io.write_point_cloud(output_path, target_pcd)
    print(f'Original voxel count: {len(source_points)}')
    print(f'Retained voxel count: {len(target_voxel_points)}')
    print(f'Filtered voxel count: {len(source_points) - len(target_voxel_points)}')
    if method == 'volume':
        print(f'Average overlap ratio: {np.mean(additional_info):.3f}')
    elif method == 'distance':
        print(f'Average distance: {np.mean(additional_info):.3f}')
    elif method == 'corner':
        print(f'Average corner count: {np.mean(additional_info):.1f}')
    return (target_voxel_points, additional_info)