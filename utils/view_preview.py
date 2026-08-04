import plotly.graph_objects as go
from utils.utils import init_3d_fig, add_cube_frame, create_voxel_sphere_union, create_optimized_outer_surface, CUBE_SIZE
from utils.core import confirmed_spheres
import numpy as np

def update_preview_plot_sphere(x: float, y: float, z: float, radius: float) -> go.Figure:
    fig = init_3d_fig()
    fig.update_layout(showlegend=False, uirevision='preview')
    from utils.core import global_mask
    if global_mask is not None:
        try:
            mask_mesh = create_optimized_outer_surface(global_mask)
            if mask_mesh is not None:
                fig.add_trace(mask_mesh)
        except Exception as e:
            print(f'Error creating mask mesh: {e}')
    mask = np.zeros(global_mask.shape, dtype=bool)
    preview_voxels = create_voxel_sphere_union(x, y, z, radius)
    if len(preview_voxels[0]) > 0:
        (x_coords, y_coords, z_coords) = preview_voxels
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        z_coords = np.array(z_coords)
        xi = np.clip(x_coords.astype(int), -1, global_mask.shape[0])
        yi = np.clip(y_coords.astype(int), -1, global_mask.shape[1])
        zi = np.clip(z_coords.astype(int), -1, global_mask.shape[2])
        mask[xi, yi, zi] = True
    sphere_mesh = create_optimized_outer_surface(mask, color='yellow')
    if sphere_mesh is not None:
        fig.add_trace(sphere_mesh)
    from utils.core import Model_input
    if Model_input is not None:
        if Model_input.dtype != bool:
            mask = Model_input > 0.5
        else:
            mask = Model_input
        input_mesh = create_optimized_outer_surface(mask, color='black')
        if input_mesh is not None:
            fig.add_trace(input_mesh)
    add_cube_frame(fig)
    return fig

def update_preview_plot_cube(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, types: str) -> go.Figure:
    fig = init_3d_fig()
    fig.update_layout(showlegend=False, uirevision='preview')
    from utils.core import global_mask
    if global_mask is not None:
        try:
            mask_mesh = create_optimized_outer_surface(global_mask)
            if mask_mesh is not None:
                fig.add_trace(mask_mesh)
        except Exception as e:
            print(f'Error creating mask mesh: {e}')
    mask = np.zeros(global_mask.shape, dtype=bool)
    if types == 'corner':
        (x_min, y_min, z_min) = (int(min(x1, x2)), int(min(y1, y2)), int(min(z1, z2)))
        (x_max, y_max, z_max) = (int(max(x1, x2)), int(max(y1, y2)), int(max(z1, z2)))
        mask[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] = True
    else:
        (x1_, y1_, z1_) = (int(x1) - int(x2), int(y1) - int(y2), int(z1) - int(z2))
        (x2_, y2_, z2_) = (int(x1) + int(x2), int(y1) + int(y2), int(z1) + int(z2))
        x1_ = np.clip(x1_, 0, CUBE_SIZE)
        y1_ = np.clip(y1_, 0, CUBE_SIZE)
        z1_ = np.clip(z1_, 0, CUBE_SIZE)
        x2_ = np.clip(x2_, 0, CUBE_SIZE)
        y2_ = np.clip(y2_, 0, CUBE_SIZE)
        z2_ = np.clip(z2_, 0, CUBE_SIZE)
        mask[x1_:x2_ + 1, y1_:y2_ + 1, z1_:z2_ + 1] = True
    cube_mesh = create_optimized_outer_surface(mask, color='yellow')
    if cube_mesh is not None:
        fig.add_trace(cube_mesh)
    from utils.core import Model_input
    if Model_input is not None:
        if Model_input.dtype != bool:
            mask = Model_input > 0.5
        else:
            mask = Model_input
        input_mesh = create_optimized_outer_surface(mask, color='black')
        if input_mesh is not None:
            fig.add_trace(input_mesh)
    add_cube_frame(fig)
    return fig