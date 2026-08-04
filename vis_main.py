import os
import json
from pathlib import Path
import argparse
import torch
import gradio as gr
import subprocess
import sys
import textwrap
from utils.utils import export_mask_ply
from utils.core import clear_core, start_new_arrow_core, confirm_arrow_core, update_handle_point_core, update_target_point_core, run_vector_field_generation_core, add_sphere_core, add_cube_core_center, add_cube_core_corner, load_voxels_ply_to_global, convert_ply_to_voxel
from utils.view_fixed import update_fixed_plot
from utils.view_vector import create_vector_field_plot
from utils.vector_get import save_lazydrag_maps_to_json, generate_explicit_correspondence_map
from utils.view_final import creat_final_plot_with_vector
global Model_path
Model_path = ''

def denormalize_xyz(sx, sy, sz):
    stz = sx
    stx = sy
    sty = -sz
    x = (stx + 0.5 - 1 / 128.0) * 64.0
    y = (sty + 0.5 - 1 / 128.0) * 64.0
    z = (stz + 0.5 - 1 / 128.0) * 64.0
    return (int(round(x)), int(round(y)), int(round(z)))

def load_data_and_preview(input_dir):
    if not input_dir or not os.path.isdir(input_dir):
        return (None, None, 'Please provide a valid working directory path.')
    glb_path = os.path.join(input_dir, 'model.glb')
    json_path = os.path.join(input_dir, 'dataset_input_clean.json')
    if not os.path.exists(glb_path):
        return (None, None, f"Model file not found: {glb_path}\nPlease ensure 'model.glb' exists in the designated directory.")
    if not os.path.exists(json_path):
        return (None, None, f"Configuration file not found: {json_path}\nPlease ensure 'dataset_input_clean.json' exists in the designated directory.")
    global Model_path
    Model_path = glb_path
    status_logs = []
    render_save_path = os.path.join(input_dir, 'render')
    os.makedirs(render_save_path, exist_ok=True)
    python_cmd_string = textwrap.dedent(f"\n        import sys\n        import os\n        sys.path.append(r'{os.path.dirname(__file__)}')\n        from inference import run_3d_rendering\n        run_3d_rendering(r'{Model_path}', r'{render_save_path}')\n    ")
    subprocess.run([sys.executable, '-c', python_cmd_string], check=True)
    VOXEL_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'voxhammer', 'delete_region_voxel.py'))
    subprocess.run([sys.executable, VOXEL_SCRIPT, '--render_dir', render_save_path, '--input_glb_path', glb_path], stdout=sys.stdout, stderr=subprocess.STDOUT, encoding='utf-8', timeout=1000)
    ply_pth = os.path.join(render_save_path, 'voxels_delete.ply')
    load_voxels_ply_to_global(ply_pth)
    status_logs.append(f'GLB voxelization executed successfully (Saved at: {render_save_path})')
    clear_core()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for sphere in data.get('confirmed_spheres', []):
        (x, y, z) = denormalize_xyz(sphere['x'], sphere['y'], sphere['z'])
        r = int(round(sphere['radius'] * 64.0))
        add_sphere_core(x, y, z, r)
    for cube in data.get('confirmed_cubes_center', []):
        (x1, y1, z1) = denormalize_xyz(cube['x1'], cube['y1'], cube['z1'])
        (x2, y2, z2) = (int(round(cube['x2'] * 64.0)), int(round(cube['y2'] * 64.0)), int(round(cube['z2'] * 64.0)))
        add_cube_core_center([x1, y1, z1], [x2, y2, z2])
    for cube in data.get('confirmed_cubes_corner', []):
        (x1, y1, z1) = denormalize_xyz(cube['x1'], cube['y1'], cube['z1'])
        (x2, y2, z2) = denormalize_xyz(cube['x2'], cube['y2'], cube['z2'])
        add_cube_core_corner([x1, y1, z1], [x2, y2, z2])
    handles = data.get('handle_points', [])
    targets = data.get('target_points', [])
    for (h, t) in zip(handles, targets):
        (hx, hy, hz) = denormalize_xyz(h[0], h[1], h[2])
        (tx, ty, tz) = denormalize_xyz(t[0], t[1], t[2])
        start_new_arrow_core()
        update_handle_point_core(hz, hy, hx)
        update_target_point_core(tz, ty, tx)
        confirm_arrow_core()
    status_logs.append('JSON structure (Mask/Control points) successfully reconstructed.')
    fixed_fig = update_fixed_plot()
    (success, msg, M_map, A_map, region_masks, _) = run_vector_field_generation_core()
    if success:
        (vec_fig, _) = create_vector_field_plot(dest=True, inpaint=True, transition=False, background=False)
        status_logs.append('Vector field visualization generated successfully.')
    else:
        vec_fig = None
        status_logs.append(f'Vector field generation warning: {msg}')
    return (fixed_fig, vec_fig, '\n'.join(status_logs))

def save_64data_wrapper(dir=''):
    from utils.core import region_masks_data, M_map_data, A_map_data, confirmed_arrows
    handle_points_list = [[a['handle'][2], a['handle'][1], a['handle'][0]] for a in confirmed_arrows]
    target_points_list = [[a['target'][2], a['target'][1], a['target'][0]] for a in confirmed_arrows]
    json_save_path = os.path.join(dir if dir else os.getcwd(), 'lazydrag_maps_data_64.json')
    save_lazydrag_maps_to_json(M_map_data, A_map_data, region_masks_data, json_save_path, handle_points=handle_points_list, target_points=target_points_list)
    return json_save_path

def save_16data_wrapper(dir=''):
    from utils.core import confirmed_arrows, global_mask, Model_input, Model_was_loaded
    handle_points_list = [[a['handle'][2] / 4, a['handle'][1] / 4, a['handle'][0] / 4] for a in confirmed_arrows]
    target_points_list = [[a['target'][2] / 4, a['target'][1] / 4, a['target'][0] / 4] for a in confirmed_arrows]
    mask_tensor = torch.from_numpy(global_mask).unsqueeze(0).unsqueeze(0).long()
    pool = torch.nn.MaxPool3d(kernel_size=4, stride=4, padding=0)
    mask_tensor = pool(mask_tensor.float()).long()
    invert_code = pool(torch.from_numpy(Model_input).unsqueeze(0).unsqueeze(0).float()).float() if Model_was_loaded else torch.zeros(1, 1, 16, 16, 16, dtype=torch.float32)
    (M_map, A_map, region_masks) = generate_explicit_correspondence_map(invert_code=invert_code, handle_points=handle_points_list, target_points=target_points_list, mask_cp_handle=mask_tensor)
    json_save_path = os.path.join(dir if dir else os.getcwd(), 'lazydrag_maps_data_16.json')
    save_lazydrag_maps_to_json(M_map, A_map, region_masks, json_save_path, handle_points_list, target_points_list)
    return json_save_path

def save_32data_wrapper(dir=''):
    from utils.core import confirmed_arrows, global_mask, Model_input, Model_was_loaded
    handle_points_list = [[a['handle'][2] / 2, a['handle'][1] / 2, a['handle'][0] / 2] for a in confirmed_arrows]
    target_points_list = [[a['target'][2] / 2, a['target'][1] / 2, a['target'][0] / 2] for a in confirmed_arrows]
    mask_tensor = torch.from_numpy(global_mask).unsqueeze(0).unsqueeze(0).long()
    pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    mask_tensor = pool(mask_tensor.float()).long()
    invert_code = pool(torch.from_numpy(Model_input).unsqueeze(0).unsqueeze(0).float()).float() if Model_was_loaded else torch.zeros(1, 1, 32, 32, 32, dtype=torch.float32)
    (M_map, A_map, region_masks) = generate_explicit_correspondence_map(invert_code=invert_code, handle_points=handle_points_list, target_points=target_points_list, mask_cp_handle=mask_tensor)
    json_save_path = os.path.join(dir if dir else os.getcwd(), 'lazydrag_maps_data_32.json')
    save_lazydrag_maps_to_json(M_map, A_map, region_masks, json_save_path, handle_points_list, target_points_list)
    return json_save_path

def run_pipeline_wrapper(is_text, input_dir, source_prompt, target_prompt, is_64, use_logit_blend):
    if not input_dir or not os.path.isdir(input_dir):
        return (None, None, 'Please provide a valid working directory path.')
    input_model = os.path.join(input_dir, 'model.glb')
    dir_all = input_dir
    mask_glb = os.path.join(dir_all, 'mask.glb')
    render_dir = os.path.join(dir_all, 'render')
    output_dir = os.path.join(dir_all, 'output')
    os.makedirs(output_dir, exist_ok=True)
    map_save_dir = dir_all
    (map_save_path2, map_save_path3) = (None, None)
    if is_64 == '64':
        map_save_path1 = save_64data_wrapper(map_save_dir)
    elif is_64 == '16':
        map_save_path1 = save_16data_wrapper(map_save_dir)
    else:
        map_save_path1 = save_16data_wrapper(map_save_dir)
        map_save_path2 = save_32data_wrapper(map_save_dir)
        map_save_path3 = save_64data_wrapper(map_save_dir)
    map_save_path = [map_save_path1, map_save_path2, map_save_path3]
    mask_ply = os.path.join(map_save_dir, 'mask.ply')
    mask_ply = export_mask_ply(json_path=map_save_path[0], out_ply=mask_ply)
    BLENDER_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'blender_ply_to_glb.py'))
    subprocess.run(['blender', '-b', '-P', BLENDER_SCRIPT, '--', mask_ply, mask_glb], stdout=sys.stdout, stderr=subprocess.STDOUT, encoding='utf-8', timeout=300)
    INFERENCE_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'inference.py'))
    output_path = os.path.join(output_dir, 'output.glb')
    cmd = [sys.executable, INFERENCE_SCRIPT, '--is_text', str(is_text), '--input_model', input_model, '--mask_glb', mask_glb, '--render_dir', render_dir, '--output_dir', output_dir, '--output_path', output_path, '--source_prompt', source_prompt, '--target_prompt', target_prompt, '--map_save_path1', map_save_path[0], '--map_save_path2', map_save_path[1], '--map_save_path3', map_save_path[2], '--is_64', is_64]
    if use_logit_blend:
        cmd.append('--use_logit_blend')
    subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT, encoding='utf-8', timeout=2000, bufsize=1, universal_newlines=True)
    ply_output_path = os.path.join(str(Path(output_path).with_suffix('')), 'st_drag_points.ply')
    convert_ply_to_voxel(ply_output_path)
    (pipeline_figure, pipeline_figure2, final_status) = creat_final_plot_with_vector(False)
    return 'Pipeline execution completed: ' + final_status
with gr.Blocks(title='Automated 3D Pipeline Workflow') as demo:
    gr.Markdown('# 3D Vector Field Pipeline (Automated Directory Parsing)')
    with gr.Row():
        input_dir = gr.Textbox(value='', label="1. Input Working Directory (Automatically parses 'model.glb' and 'dataset_input_clean.json')", placeholder='e.g., /path/to/your/workspace')
    with gr.Row():
        parse_btn = gr.Button('2. Parse Directory, Reconstruct Mask, and Generate Preview', variant='primary', size='lg')
    with gr.Row():
        status_text = gr.Textbox(label='Status Logs', value='Awaiting operation...', lines=4, interactive=False)
    with gr.Row():
        fixed_plot = gr.Plot(label='View 1: GLB Model and Control Points (HP/TP) Restoration')
        vector_field_plot = gr.Plot(label='View 2: Vector Field Distribution Visualization')
    gr.Markdown('---')
    gr.Markdown('### Pipeline Operational Parameters')
    with gr.Row():
        with gr.Column(scale=1):
            is_text_radio = gr.Radio(choices=[True], value=True, label='Driving Modality (True: Text | False: Image)')
            is_64 = gr.Radio(choices=['64+32+16'], value='64+32+16', label='Vector Field Resolution')
            use_logit_blend = gr.Radio(choices=[True], value=True, label='Apply Logit Blend')
        with gr.Column(scale=2):
            source_prompt = gr.Textbox(value='A toy car', label='Source Prompt (Text-driven)')
            target_prompt = gr.Textbox(value='A toy car', label='Target Prompt (Text-driven)')
    run_pipeline_btn = gr.Button('3. Execute End-to-End Pipeline', variant='stop', size='lg')
    parse_btn.click(fn=load_data_and_preview, inputs=[input_dir], outputs=[fixed_plot, vector_field_plot, status_text])
    run_pipeline_btn.click(fn=run_pipeline_wrapper, inputs=[is_text_radio, input_dir, source_prompt, target_prompt, is_64, use_logit_blend], outputs=[status_text])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated Directory-Based 3D Pipeline UI')
    parser.add_argument('--server_port', type=int, default=1879, help='Port number')
    args = parser.parse_args()
    demo.launch(server_port=args.server_port, share=False, show_error=True)