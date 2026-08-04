import plotly.graph_objects as go
import torch
import numpy as np
from utils.utils import init_3d_fig, add_cube_frame, create_arrow_with_shaft, create_optimized_outer_surface
from typing import Tuple

def create_vector_field_plot(dest=True, inpaint=True, transition=True, background=True) -> Tuple[go.Figure, str]:
    from utils.core import confirmed_spheres, vector_field_data, region_masks_data, M_map_data, A_map_data, vector_display_percentage, CUBE_SIZE
    if vector_field_data is None or region_masks_data is None:
        fig = init_3d_fig()
        return (fig, "❌ Please click the 'Run' button first to generate vector field data")
    fig = init_3d_fig()
    fig.update_layout(showlegend=True, uirevision='vector_field', title=f'Vector Field Visualization (Displaying {vector_display_percentage}% of arrows)')
    region_colors = {'destination': {0: 'rgba(0, 0, 255, 0.3)', 1: 'rgba(255, 0, 0, 0.3)', 2: 'rgba(128, 0, 128, 0.3)', 3: 'rgba(255, 165, 0, 0.3)'}, 'inpainting': 'rgba(255, 255, 0, 0.3)', 'transition': 'rgba(0, 255, 0, 0.3)', 'background': 'rgba(128, 128, 128, 0.3)'}
    update_mask_view(fig, dest, inpaint, transition, background, region_colors)
    if M_map_data is not None and A_map_data is not None:
        valid_positions = torch.nonzero(A_map_data > 0)
        total_valid = len(valid_positions)
        if total_valid > 0:
            num_arrows = min(max(1, int(total_valid * vector_display_percentage / 100)), 100)
            indices = torch.randperm(total_valid)[:num_arrows]
            selected_positions = valid_positions[indices]
            print(f'Debug: Displaying {num_arrows} arrows out of {total_valid} valid positions')
            for (i, pos) in enumerate(selected_positions):
                if len(pos) == 4:
                    (_, z, y, x) = pos.tolist()
                else:
                    (z, y, x) = pos.tolist()
                displacement = M_map_data[z, y, x]
                (src_z, src_y, src_x) = displacement.tolist()
                start_point = [src_x, src_y, src_z]
                end_point = [x, y, z]
                direction = np.array(end_point) - np.array(start_point)
                length = np.linalg.norm(direction)
                if length > 0:
                    direction_normalized = direction / length
                    display_start_point = start_point
                    display_end_point = end_point
                    arrow_color = 'rgba(255, 100, 0, 0.8)'
                    arrow_components = create_arrow_with_shaft(display_start_point, display_end_point, color=arrow_color, name=f'vector_{i}')
                    for component in arrow_components:
                        fig.add_trace(component)
    add_cube_frame(fig)
    return (fig, f'✅ Vector field generation complete! Displaying {vector_display_percentage}% of arrow samples')

def update_mask_view(fig, dest, inpaint, transition, background, region_colors=None):
    from utils.core import region_masks_data
    if region_masks_data is None:
        return (fig, "❌ Please click the 'Run' button first to generate vector field data")
    else:
        if dest:
            from utils.vector_get import destination_mask_dict_use
            if destination_mask_dict_use != None:
                destination_mask_list = destination_mask_dict_use
                for (idx, mask) in enumerate(destination_mask_list):
                    color = region_colors.get('destination', {}).get(idx, 'rgba(0,0,255,0.3)')
                    dest_mesh = create_optimized_outer_surface(mask, color=color)
                    if dest_mesh is not None:
                        fig.add_trace(dest_mesh)
        if inpaint:
            inpaint_mask = region_masks_data.get('inpainting', None)
            inpaint_mesh = create_optimized_outer_surface(inpaint_mask, color=region_colors.get('inpainting', 'yellow'))
            if inpaint_mesh is not None:
                fig.add_trace(inpaint_mesh)
        if transition:
            transition_mask = region_masks_data.get('transition', None)
            transition_mesh = create_optimized_outer_surface(transition_mask, color=region_colors.get('transition', 'green'))
            if transition_mesh is not None:
                fig.add_trace(transition_mesh)
        if background:
            background_mask = region_masks_data.get('background', None)
            background_mesh = create_optimized_outer_surface(background_mask, color=region_colors.get('background', 'gray'))
            if background_mesh is not None:
                fig.add_trace(background_mesh)
    return (fig, '✅ Region mask display update complete!')