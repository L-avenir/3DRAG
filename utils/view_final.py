import plotly.graph_objects as go
import numpy as np
from utils.utils import init_3d_fig, add_cube_frame, create_optimized_outer_surface
from utils.view_vector import create_vector_field_plot
global g_fig_1, g_fig_2
g_fig_1 = None
import plotly.express as px

def creat_final_plot_with_vector(if_add_vector=False) -> go.Figure:
    global g_fig_1, g_fig_2
    from utils.core import Model_output, Model_input
    if if_add_vector:
        (fig, info) = create_vector_field_plot(True, True, False, False)
        (fig2, _) = create_vector_field_plot(True, True, False, False)
    else:
        fig = init_3d_fig()
        fig.update_layout(showlegend=False, uirevision='fixed')
        fig2 = init_3d_fig()
        fig2.update_layout(showlegend=False, uirevision='fixed')
        info = ''
    if Model_output is not None:
        if Model_output.dtype != bool:
            mask = Model_output > 0.5
        else:
            mask = Model_output
        input_mesh = create_optimized_outer_surface(mask)
        if input_mesh is not None:
            fig.add_trace(input_mesh)
    if Model_input is not None:
        if Model_input.dtype != bool:
            mask = Model_input > 0.5
        else:
            mask = Model_input
        input_mesh2 = create_optimized_outer_surface(mask)
        if input_mesh2 is not None:
            fig2.add_trace(input_mesh2)
    add_cube_frame(fig)
    add_cube_frame(fig2)
    info += '  Finish the visualize of final plot'
    g_fig_1 = fig
    return (fig, fig2, info)

def creat_sectional_view_with_vector(plane_type, slider_position):
    global g_fig_1
    info = ''
    try:
        if g_fig_1 is None or len(g_fig_1.data) == 0:
            raise ValueError('Global 3D fig is empty, no voxel data to parse')
        volume_data = np.zeros((64, 64, 64), dtype=np.float32)
        for trace in g_fig_1.data:
            if trace.type == 'mesh3d' and hasattr(trace, 'x') and hasattr(trace, 'y') and hasattr(trace, 'z'):
                points = np.column_stack([trace.x, trace.y, trace.z])
                points = ((points + 1) / 2 * 63).astype(int)
                for (x, y, z) in points:
                    if 0 <= x < 64 and 0 <= y < 64 and (0 <= z < 64):
                        volume_data[z, y, x] = 1.0
                break
        (z_len, y_len, x_len) = (64, 64, 64)
        info += f'Data shape: ({z_len}, {y_len}, {x_len}) | '
    except Exception as e:
        print(f'Warning: Could not extract volume data: {e}')
        volume_data = None
        z_len = y_len = x_len = 64

    def map_slider_to_coord(slider_val, max_dim, coord_min=-1, coord_max=1):
        normalized = slider_val / 63.0
        return coord_min + normalized * (coord_max - coord_min)
    plane_params = {}
    slice_idx = -1
    if plane_type == 'XY Plane':
        z_coord = map_slider_to_coord(slider_position, z_len, -1, 1)
        slice_idx = int((z_coord + 1) / 2 * (z_len - 1))
        plane_params = {'type': 'XY Plane', 'normal': (0, 0, 1), 'point': (0, 0, z_coord), 'range_x': (-1, 1), 'range_y': (-1, 1), 'slice_axis': 'z', 'slice_value': z_coord, 'slice_idx': slice_idx}
        info += f'XY Plane at Z={z_coord:.3f} (index={slice_idx}) | '
    elif plane_type == 'YZ Plane':
        x_coord = map_slider_to_coord(slider_position, x_len, -1, 1)
        slice_idx = int((x_coord + 1) / 2 * (x_len - 1))
        plane_params = {'type': 'YZ Plane', 'normal': (1, 0, 0), 'point': (x_coord, 0, 0), 'range_y': (-1, 1), 'range_z': (-1, 1), 'slice_axis': 'x', 'slice_value': x_coord, 'slice_idx': slice_idx}
        info += f'YZ Plane at X={x_coord:.3f} (index={slice_idx}) | '
    elif plane_type == 'XZ Plane':
        y_coord = map_slider_to_coord(slider_position, y_len, -1, 1)
        slice_idx = int((y_coord + 1) / 2 * (y_len - 1))
        plane_params = {'type': 'XZ Plane', 'normal': (0, 1, 0), 'point': (0, y_coord, 0), 'range_x': (-1, 1), 'range_z': (-1, 1), 'slice_axis': 'y', 'slice_value': y_coord, 'slice_idx': slice_idx}
        info += f'XZ Plane at Y={y_coord:.3f} (index={slice_idx}) | '
    try:
        slice_idx = np.clip(slice_idx, 0, 63)
        slice_2d = None
        axis_labels = {'x': '', 'y': '', 'title': ''}
        if plane_type == 'XY Plane' and volume_data is not None:
            slice_2d = volume_data[slice_idx, :, :]
            axis_labels = {'x': 'Y-axis (voxels)', 'y': 'X-axis (voxels)', 'title': f'XY Plane Section (Z={slice_idx})'}
        elif plane_type == 'YZ Plane' and volume_data is not None:
            slice_2d = volume_data[:, :, slice_idx]
            axis_labels = {'x': 'Y-axis (voxels)', 'y': 'Z-axis (voxels)', 'title': f'YZ Plane Section (X={slice_idx})'}
        elif plane_type == 'XZ Plane' and volume_data is not None:
            slice_2d = volume_data[:, slice_idx, :]
            axis_labels = {'x': 'X-axis (voxels)', 'y': 'Z-axis (voxels)', 'title': f'XZ Plane Section (Y={slice_idx})'}
        if slice_2d is not None:
            slice_2d_binary = (slice_2d > 0.5).astype(np.float32)
            fig_2d = px.imshow(slice_2d_binary, x=np.arange(64), y=np.arange(64), labels=dict(x=axis_labels['x'], y=axis_labels['y'], color='Voxel presence'), title=axis_labels['title'], color_continuous_scale=['white', 'black'], range_color=[0, 1], width=600, height=600)
            fig_2d.update_layout(title_font=dict(size=16), coloraxis_colorbar=dict(title='Voxel status', tickvals=[0, 1], ticktext=['Absent', 'Present'], orientation='h', x=0.5, y=-0.15), xaxis=dict(tickmode='linear', tick0=0, dtick=8, showgrid=True), yaxis=dict(tickmode='linear', tick0=0, dtick=8, showgrid=True), plot_bgcolor='white')
            fig_sectional = fig_2d
            info += '2D section generated successfully | '
        else:
            fig_sectional = go.Figure()
            fig_sectional.add_annotation(text='Cannot generate section (no voxel data)', x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14, color='red'))
            fig_sectional.update_layout(title='Section generation failed', xaxis=dict(visible=False), yaxis=dict(visible=False), width=600, height=600)
            info += '2D section generation failed (no data) | '
    except Exception as e:
        print(f'Error generating 2D section: {e}')
        fig_sectional = go.Figure()
        fig_sectional.add_annotation(text=f'Section generation error: {str(e)}', x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14, color='red'))
        fig_sectional.update_layout(title='Section generation failed', xaxis=dict(visible=False), yaxis=dict(visible=False), width=600, height=600)
        info += f'2D section generation error: {str(e)} | '
    info += 'Sectional view created successfully'
    return (fig_sectional, info)