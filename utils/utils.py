import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Any
CUBE_SIZE = 63
MASK_SHAPE = (64, 64, 64)
VOXEL_SIZE = 1.0
VOXEL_DENSITY = 1

def extract_outer_surface(mask):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if mask.size == 0:
        return None
    boundary_mask = np.zeros_like(mask, dtype=bool)
    (depth, height, width) = mask.shape
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if mask[z, y, x]:
                    is_boundary = False
                    for (dz, dy, dx) in directions:
                        (nz, ny, nx) = (z + dz, y + dy, x + dx)
                        if nz < 0 or nz >= depth or ny < 0 or (ny >= height) or (nx < 0) or (nx >= width) or (not mask[nz, ny, nx]):
                            is_boundary = True
                            break
                    if is_boundary:
                        boundary_mask[z, y, x] = True
    return boundary_mask

def create_outer_surface_mesh(mask):
    boundary_mask = extract_outer_surface(mask)
    if not np.any(boundary_mask):
        return None
    all_vertices = []
    all_triangles = []
    vertex_offset = 0
    cube_vertices = np.array([[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]])
    cube_faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    for idx in np.ndindex(mask.shape):
        if boundary_mask[idx]:
            (z, y, x) = idx
            vertices = cube_vertices + np.array([x, y, z])
            all_vertices.extend(vertices.tolist())
            for face in cube_faces:
                adjusted_face = [v + vertex_offset for v in face]
                all_triangles.append(adjusted_face)
            vertex_offset += len(vertices)
    if not all_vertices:
        return None
    mesh = go.Mesh3d(x=[v[0] for v in all_vertices], y=[v[1] for v in all_vertices], z=[v[2] for v in all_vertices], i=[f[0] for f in all_triangles], j=[f[1] for f in all_triangles], k=[f[2] for f in all_triangles], opacity=0.8, color='gray', flatshading=True, name='outer_surface_centered')
    return mesh

def create_optimized_outer_surface(mask, color='lightblue') -> Any:
    boundary_mask = extract_outer_surface(mask)
    if not np.any(boundary_mask):
        return None
    faces_data = {'top': [], 'bottom': [], 'front': [], 'back': [], 'left': [], 'right': []}
    all_vertices = []
    vertex_map = {}
    current_vertex_id = 0

    def get_vertex_id(x, y, z):
        nonlocal current_vertex_id
        key = (x, y, z)
        if key not in vertex_map:
            vertex_map[key] = current_vertex_id
            all_vertices.append([x, y, z])
            current_vertex_id += 1
        return vertex_map[key]
    face_directions = {'top': (0, 0, 1), 'bottom': (0, 0, -1), 'front': (0, -1, 0), 'back': (0, 1, 0), 'left': (-1, 0, 0), 'right': (1, 0, 0)}
    for idx in np.ndindex(mask.shape):
        if boundary_mask[idx]:
            (z, y, x) = idx
            (x_min, x_max) = (x - 0.5, x + 0.5)
            (y_min, y_max) = (y - 0.5, y + 0.5)
            (z_min, z_max) = (z - 0.5, z + 0.5)
            for (face_name, (dx, dy, dz)) in face_directions.items():
                (nz, ny, nx) = (z + dz, y + dy, x + dx)
                show_face = False
                if nz < 0 or nz >= mask.shape[0] or ny < 0 or (ny >= mask.shape[1]) or (nx < 0) or (nx >= mask.shape[2]) or (not mask[nz, ny, nx]):
                    show_face = True
                if show_face:
                    if face_name == 'bottom':
                        v0 = get_vertex_id(x_min, y_min, z_min)
                        v1 = get_vertex_id(x_max, y_min, z_min)
                        v2 = get_vertex_id(x_max, y_max, z_min)
                        v3 = get_vertex_id(x_min, y_max, z_min)
                        faces_data['bottom'].extend([[v0, v1, v2], [v0, v2, v3]])
                    elif face_name == 'top':
                        v0 = get_vertex_id(x_min, y_min, z_max)
                        v1 = get_vertex_id(x_max, y_min, z_max)
                        v2 = get_vertex_id(x_max, y_max, z_max)
                        v3 = get_vertex_id(x_min, y_max, z_max)
                        faces_data['top'].extend([[v0, v2, v1], [v0, v3, v2]])
                    elif face_name == 'front':
                        v0 = get_vertex_id(x_min, y_min, z_min)
                        v1 = get_vertex_id(x_max, y_min, z_min)
                        v2 = get_vertex_id(x_max, y_min, z_max)
                        v3 = get_vertex_id(x_min, y_min, z_max)
                        faces_data['front'].extend([[v0, v2, v1], [v0, v3, v2]])
                    elif face_name == 'back':
                        v0 = get_vertex_id(x_min, y_max, z_min)
                        v1 = get_vertex_id(x_max, y_max, z_min)
                        v2 = get_vertex_id(x_max, y_max, z_max)
                        v3 = get_vertex_id(x_min, y_max, z_max)
                        faces_data['back'].extend([[v0, v1, v2], [v0, v2, v3]])
                    elif face_name == 'left':
                        v0 = get_vertex_id(x_min, y_min, z_min)
                        v1 = get_vertex_id(x_min, y_max, z_min)
                        v2 = get_vertex_id(x_min, y_max, z_max)
                        v3 = get_vertex_id(x_min, y_min, z_max)
                        faces_data['left'].extend([[v0, v1, v2], [v0, v2, v3]])
                    elif face_name == 'right':
                        v0 = get_vertex_id(x_max, y_min, z_min)
                        v1 = get_vertex_id(x_max, y_max, z_min)
                        v2 = get_vertex_id(x_max, y_max, z_max)
                        v3 = get_vertex_id(x_max, y_min, z_max)
                        faces_data['right'].extend([[v0, v2, v1], [v0, v3, v2]])
    all_triangles = []
    for face_triangles in faces_data.values():
        all_triangles.extend(face_triangles)
    if not all_vertices:
        return None
    mesh = go.Mesh3d(x=[v[0] for v in all_vertices], y=[v[1] for v in all_vertices], z=[v[2] for v in all_vertices], i=[tri[0] for tri in all_triangles], j=[tri[1] for tri in all_triangles], k=[tri[2] for tri in all_triangles], opacity=0.4, color=color, flatshading=True)
    return mesh

def create_voxel_sphere_union(x0, y0, z0, radius) -> Tuple[List[int], List[int], List[int]]:
    min_x = max(0, int(np.floor(x0 - radius)))
    max_x = min(CUBE_SIZE, int(np.ceil(x0 + radius)))
    min_y = max(0, int(np.floor(y0 - radius)))
    max_y = min(CUBE_SIZE, int(np.ceil(y0 + radius)))
    min_z = max(0, int(np.floor(z0 - radius)))
    max_z = min(CUBE_SIZE, int(np.ceil(z0 + radius)))
    (voxels_x, voxels_y, voxels_z) = ([], [], [])
    for x in range(min_x, max_x + 1, VOXEL_DENSITY):
        for y in range(min_y, max_y + 1, VOXEL_DENSITY):
            for z in range(min_z, max_z + 1, VOXEL_DENSITY):
                dist_sq = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
                if dist_sq <= radius ** 2:
                    voxels_x.append(x)
                    voxels_y.append(y)
                    voxels_z.append(z)
    return (voxels_x, voxels_y, voxels_z)
import numpy as np
import plotly.graph_objects as go
from typing import List, Any

def create_arrow_with_shaft(start_point: List[int], end_point: List[int], color='red', name='arrow') -> List[Any]:
    (x0, y0, z0) = map(float, start_point)
    (x1, y1, z1) = map(float, end_point)
    (dx, dy, dz) = (x1 - x0, y1 - y0, z1 - z0)
    length = np.linalg.norm([dx, dy, dz])
    if length < 0.001:
        return []
    (dx_unit, dy_unit, dz_unit) = (dx / length, dy / length, dz / length)
    arrows_list = []
    eps = 1e-06
    shaft_segments = 8
    shaft_radius_ratio = 0.02
    head_radius_ratio = 0.05
    head_length_ratio = 0.1
    shaft_radius = length * shaft_radius_ratio
    head_radius = length * head_radius_ratio
    head_length = length * head_length_ratio
    shaft_radius = max(shaft_radius, 0.08)
    head_radius = max(head_radius, 0.2)
    head_length = max(head_length, 0.4)
    shaft_radius = min(shaft_radius, 0.6)
    head_radius = min(head_radius, 1.0)
    head_length = min(head_length, 2.0)
    shaft_points = []
    head_base_center = [x1 - head_length * dx_unit, y1 - head_length * dy_unit, z1 - head_length * dz_unit]
    for i in range(shaft_segments + 1):
        t = i / shaft_segments
        px = x0 + t * (head_base_center[0] - x0)
        py = y0 + t * (head_base_center[1] - y0)
        pz = z0 + t * (head_base_center[2] - z0)
        shaft_points.append([px, py, pz])
    (shaft_x, shaft_y, shaft_z) = ([], [], [])
    for point in shaft_points:
        (cx, cy, cz) = point
        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            if abs(dx_unit) < 0.9 - eps:
                perp1 = np.array([-dy_unit, dx_unit, 0.0])
            else:
                perp1 = np.array([0.0, -dz_unit, dy_unit])
            perp1_norm = np.linalg.norm(perp1)
            if perp1_norm < eps:
                perp1 = np.array([1.0, 0.0, 0.0]) if abs(dx_unit) < eps else np.array([0.0, 1.0, 0.0])
                perp1_norm = np.linalg.norm(perp1)
            perp1 = perp1 / perp1_norm
            perp2 = np.cross([dx_unit, dy_unit, dz_unit], perp1)
            perp2_norm = np.linalg.norm(perp2)
            if perp2_norm < eps:
                perp2 = np.array([0.0, 0.0, 1.0])
                perp2_norm = np.linalg.norm(perp2)
            perp2 = perp2 / perp2_norm
            circle_x = cx + shaft_radius * (perp1[0] * np.cos(angle) + perp2[0] * np.sin(angle))
            circle_y = cy + shaft_radius * (perp1[1] * np.cos(angle) + perp2[1] * np.sin(angle))
            circle_z = cz + shaft_radius * (perp1[2] * np.cos(angle) + perp2[2] * np.sin(angle))
            shaft_x.append(circle_x)
            shaft_y.append(circle_y)
            shaft_z.append(circle_z)
    if len(shaft_x) > 0:
        shaft_mesh = go.Mesh3d(x=shaft_x, y=shaft_y, z=shaft_z, color=color, opacity=0.6, name=f'{name}_shaft', showscale=False)
        arrows_list.append(shaft_mesh)
    (head_base_x, head_base_y, head_base_z) = ([], [], [])
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        if abs(dx_unit) < 0.9 - eps:
            perp1 = np.array([-dy_unit, dx_unit, 0.0])
        else:
            perp1 = np.array([0.0, -dz_unit, dy_unit])
        perp1_norm = np.linalg.norm(perp1)
        if perp1_norm < eps:
            perp1 = np.array([1.0, 0.0, 0.0]) if abs(dx_unit) < eps else np.array([0.0, 1.0, 0.0])
            perp1_norm = np.linalg.norm(perp1)
        perp1 = perp1 / perp1_norm
        perp2 = np.cross([dx_unit, dy_unit, dz_unit], perp1)
        perp2_norm = np.linalg.norm(perp2)
        if perp2_norm < eps:
            perp2 = np.array([0.0, 0.0, 1.0])
            perp2_norm = np.linalg.norm(perp2)
        perp2 = perp2 / perp2_norm
        base_x = head_base_center[0] + head_radius * (perp1[0] * np.cos(angle) + perp2[0] * np.sin(angle))
        base_y = head_base_center[1] + head_radius * (perp1[1] * np.cos(angle) + perp2[1] * np.sin(angle))
        base_z = head_base_center[2] + head_radius * (perp1[2] * np.cos(angle) + perp2[2] * np.sin(angle))
        head_base_x.append(base_x)
        head_base_y.append(base_y)
        head_base_z.append(base_z)
    (head_i, head_j, head_k) = ([], [], [])
    tip_idx = 0
    for i in range(len(head_base_x)):
        base_idx = i + 1
        next_base_idx = (i + 1) % len(head_base_x) + 1
        head_i.append(tip_idx)
        head_j.append(base_idx)
        head_k.append(next_base_idx)
    if len(head_base_x) > 0:
        head_mesh = go.Mesh3d(x=[x1] + head_base_x, y=[y1] + head_base_y, z=[z1] + head_base_z, i=head_i, j=head_j, k=head_k, color=color, opacity=0.8, name=f'{name}_head', showscale=False)
        arrows_list.append(head_mesh)
    center_line = go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines', line=dict(color=color, width=1), showlegend=False, name=f'{name}_center')
    arrows_list.append(center_line)
    return arrows_list

def add_cube_frame(fig: go.Figure) -> None:
    cube_vertices = np.array([[0, 0, 0], [0, 0, CUBE_SIZE], [0, CUBE_SIZE, 0], [0, CUBE_SIZE, CUBE_SIZE], [CUBE_SIZE, 0, 0], [CUBE_SIZE, 0, CUBE_SIZE], [CUBE_SIZE, CUBE_SIZE, 0], [CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for edge in edges:
        (start, end) = edge
        fig.add_trace(go.Scatter3d(x=[cube_vertices[start][0], cube_vertices[end][0]], y=[cube_vertices[start][1], cube_vertices[end][1]], z=[cube_vertices[start][2], cube_vertices[end][2]], mode='lines', line=dict(color='gray', width=2), showlegend=False))

def init_3d_fig(layout_width=800, layout_height=700) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(width=layout_width, height=layout_height, margin=dict(l=0, r=0, b=0, t=20), scene=dict(xaxis=dict(range=[0, CUBE_SIZE], title='X Voxel', backgroundcolor='white', dtick=8), yaxis=dict(range=[0, CUBE_SIZE], title='Y Voxel', backgroundcolor='white', dtick=8), zaxis=dict(range=[0, CUBE_SIZE], title='Z Voxel (Vertical)', backgroundcolor='white', dtick=8), aspectmode='cube', camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.8, y=1.8, z=1.8), center=dict(x=0, y=0, z=0)), dragmode='orbit', camera_projection_type='orthographic'))
    return fig

import trimesh
import json

def mask_zyx_to_mesh(mask_zyx: np.ndarray) -> trimesh.Trimesh:
    assert mask_zyx.ndim == 3
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0)).astype(bool)
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(mask_xyz)
    mesh.apply_scale(1.0 / 64.0)
    mesh.apply_translation([-0.5, -0.5, -0.5])
    if not mesh.is_watertight:
        pass
    mesh.process(validate=True)
    return mesh

def export_mask_ply(json_path: str, out_ply: str):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    rm = data['region_masks']
    dest = np.array(rm['destination'], dtype=bool)
    inp = np.array(rm['inpainting'], dtype=bool)
    mask = dest | inp
    mesh = mask_zyx_to_mesh(mask)
    mesh.export(out_ply)
    print(f'[OK] wrote: {out_ply}  verts={len(mesh.vertices)} faces={len(mesh.faces)}')
    return out_ply