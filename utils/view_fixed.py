import plotly.graph_objects as go
from utils.utils import init_3d_fig, add_cube_frame, create_arrow_with_shaft, create_optimized_outer_surface

def update_fixed_plot() -> go.Figure:
    from utils.core import global_mask, confirmed_arrows, current_handle_point, current_target_point, has_unsaved_arrow
    fig = init_3d_fig()
    fig.update_layout(showlegend=False, uirevision='fixed')
    if global_mask is not None:
        try:
            mask_mesh = create_optimized_outer_surface(global_mask)
            if mask_mesh is not None:
                fig.add_trace(mask_mesh)
        except Exception as e:
            print(f'Error creating mask mesh: {e}')
    for (i, arrow) in enumerate(confirmed_arrows):
        handle = arrow['handle']
        target = arrow['target']
        fig.add_trace(go.Scatter3d(x=[handle[0]], y=[handle[1]], z=[handle[2]], mode='markers', marker=dict(size=4, color='green', opacity=1.0, symbol='circle'), showlegend=False, hoverinfo='text', text=f'Arrow {i + 1} start: ({handle[0]},{handle[1]},{handle[2]})'))
        fig.add_trace(go.Scatter3d(x=[target[0]], y=[target[1]], z=[target[2]], mode='markers', marker=dict(size=4, color='blue', opacity=1.0, symbol='diamond'), showlegend=False, hoverinfo='text', text=f'Arrow {i + 1} end: ({target[0]},{target[1]},{target[2]})'))
        arrow_components = create_arrow_with_shaft(handle, target, color='red', name=f'Arrow {i + 1}')
        for component in arrow_components:
            fig.add_trace(component)
    if has_unsaved_arrow:
        fig.add_trace(go.Scatter3d(x=[current_handle_point[0]], y=[current_handle_point[1]], z=[current_handle_point[2]], mode='markers', marker=dict(size=5, color='lime', opacity=1.0, symbol='circle', line=dict(color='black', width=2)), showlegend=False, hoverinfo='text', text=f'Unsaved start: ({current_handle_point[0]},{current_handle_point[1]},{current_handle_point[2]})'))
        fig.add_trace(go.Scatter3d(x=[current_target_point[0]], y=[current_target_point[1]], z=[current_target_point[2]], mode='markers', marker=dict(size=5, color='cyan', opacity=1.0, symbol='diamond', line=dict(color='black', width=2)), showlegend=False, hoverinfo='text', text=f'Unsaved end: ({current_target_point[0]},{current_target_point[1]},{current_target_point[2]})'))
        arrow_components = create_arrow_with_shaft(current_handle_point, current_target_point, color='orange', name='Unsaved arrow')
        for component in arrow_components:
            fig.add_trace(component)
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