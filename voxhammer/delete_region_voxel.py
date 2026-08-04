import os
import bpy
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from voxhammer.util_voxel_filtering import process_voxels_with_improved_filtering
import argparse

def glb_to_ply(input_glb_path, input_ply_path):
    print('[DBG] Entered the glb_to_ply function')
    if not os.path.exists(input_glb_path):
        raise FileNotFoundError(f'Input GLB file not found: {input_glb_path}')
    print(f'[DBG] Input GLB path: {input_glb_path}')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.import_scene.gltf(filepath=input_glb_path)
    bpy.ops.wm.ply_export(filepath=input_ply_path, export_normals=True, ascii_format=True)
    print(f'[DBG] Successfully exported PLY file: {input_ply_path}')

def process_delete_ply(input_glb_path, render_dir, filter_method='volume', voxel_size=1 / 64):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.append(root_path)
    preset_voxel_path = f'{root_path}/assets/preset/preset_grid64.ply'
    print(input_glb_path, render_dir, filter_method, voxel_size)
    input_ply_path = os.path.join(render_dir, 'mesh_delete.ply')
    output_ply_path = os.path.join(render_dir, 'voxels_delete.ply')
    glb_to_ply(input_glb_path, input_ply_path)
    (outside_voxel_points, additional_info) = process_voxels_with_improved_filtering(preset_voxel_path, input_ply_path, output_ply_path, method=filter_method, voxel_size=voxel_size, inside=True)

def main():
    parser = argparse.ArgumentParser(description='Independently execute the 3D editing pipeline, accepting Gradio arguments')
    parser.add_argument('--render_dir', type=str, required=True)
    parser.add_argument('--input_glb_path', type=str, required=True)
    parser.add_argument('--filter_method', type=str, default='distance', choices=['volume', 'distance', 'corner'], help='Filter method (default: distance)')
    parser.add_argument('--voxel_size', type=float, default=1 / 64, help='Voxel size (default: 1/64)')
    args = parser.parse_args()
    process_delete_ply(args.input_glb_path, args.render_dir, args.filter_method, args.voxel_size)

if __name__ == '__main__':
    main()