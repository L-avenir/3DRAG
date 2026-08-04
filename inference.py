import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import sys
import argparse
from typing import Optional
from voxhammer.edit_pipeline_preposemb import run_edit
from voxhammer.bpy_render import render_3d_model
from voxhammer.extract_feature import extract_features
from voxhammer.delete_region_voxel import process_delete_ply
from trellis.pipelines import TrellisTextTo3DPipeline
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)
from utils.map_to_std_type import map_to_std, map_to_std_16, map_to_std_16_with_32_and_64
from trellis.pipelines import TrellisTextTo3DPipeline
global DIR_PATH
DIR_PATH = root_path

def run_3d_rendering(input_model_path: str, render_dir: str, **render_kwargs) -> dict:
    print('=' * 50)
    print('STEP 1: 3D Model Rendering')
    print('=' * 50)
    if os.path.exists(os.path.join(render_dir, 'transforms.json')) and os.path.exists(os.path.join(render_dir, 'mesh.ply')):
        print(f'Render directory {render_dir} already exists')
        return {'rendered': True, 'num_views': 150, 'output_dir': render_dir, 'transforms_file': os.path.join(render_dir, 'transforms.json'), 'mesh_file': os.path.join(render_dir, 'mesh.ply')}
    default_params = {'num_views': 150, 'scale': 1.0, 'offset': None, 'resolution': 512, 'engine': 'CYCLES', 'geo_mode': False, 'split_normal': False, 'save_mesh': True}
    default_params.update(render_kwargs)
    print(f'Input model: {input_model_path}')
    print(f'Output directory: {render_dir}')
    print(f'Rendering parameters: {default_params}')
    result = render_3d_model(file_path=input_model_path, output_dir=render_dir, **default_params)
    print(f'Rendering completed successfully!')
    print(f"Generated {result['num_views']} views")
    print(f"Transforms file: {result['transforms_file']}")
    if result['mesh_file']:
        print(f"Mesh file: {result['mesh_file']}")
    return result

def run_feature_extraction(render_dir: str, **feature_kwargs) -> dict:
    print('=' * 50)
    print('STEP 2: Feature Extraction')
    print('=' * 50)
    default_params = {'model': 'dinov2_vitl14_reg', 'batch_size': 10}
    default_params.update(feature_kwargs)
    print(f'Render directory: {render_dir}')
    print(f'Feature extraction parameters: {default_params}')
    extract_features(render_dir, **default_params)
    features_path = os.path.join(render_dir, 'features.npz')
    print(f'Feature extraction completed successfully!')
    print(f'Features saved to: {features_path}')
    return {'features_path': features_path}

def run_voxel_masking(mask_glb_path: str, render_dir: str, **mask_kwargs) -> dict:
    print('=' * 50)
    print('STEP 3: Voxel Masking')
    print('=' * 50)
    default_params = {'filter_method': 'volume', 'voxel_size': 1 / 64}
    default_params.update(mask_kwargs)
    print(f'Mask GLB file: {mask_glb_path}')
    print(f'Render directory: {render_dir}')
    print(f'Masking parameters: {default_params}')
    process_delete_ply(mask_glb_path, render_dir, **default_params)
    voxels_delete_path = os.path.join(render_dir, 'voxels_delete.ply')
    print(f'Voxel masking completed successfully!')
    print(f'Mask file: {voxels_delete_path}')
    return {'mask_path': voxels_delete_path}

def run_3d_editing(pipeline, render_dir: str, output_path: str, image_dir: str, is_text: bool, source_prompt: str, target_prompt: str, **edit_kwargs) -> dict:
    print('=' * 50)
    print('STEP 4: 3D Editing')
    print('=' * 50)
    default_params = {'skip_step': 0, 're_init': False, 'cfg': [5.0, 6.0, 0.0, 0.0]}
    default_params.update(edit_kwargs)
    print(f'Render directory: {render_dir}')
    print(f'Image directory: {image_dir}')
    print(f'Output path: {output_path}')
    print(f'Editing parameters: {default_params}')
    required_files = [os.path.join(render_dir, 'voxels.ply'), os.path.join(render_dir, 'features.npz'), os.path.join(render_dir, 'voxels_delete.ply')]
    if not is_text:
        required_files.extend([os.path.join(image_dir, '2d_render.png'), os.path.join(image_dir, '2d_edit.png'), os.path.join(image_dir, '2d_mask.png')])
    try:
        run_edit(pipeline, render_dir, output_path, image_dir, is_text, source_prompt, target_prompt, **default_params)
        print(f'3D editing completed successfully!')
        print(f'Final result saved to: {output_path}')
        return {'output_path': output_path}
    except Exception as e:
        print(f'3D editing failed: {e}')
        raise

def run_complete_pipeline(pipeline, input_model_path: str, mask_glb_path: str, render_dir: str, output_path: str, image_dir: str, is_text: bool, source_prompt: str, target_prompt: str, render_params: Optional[dict]=None, feature_params: Optional[dict]=None, mask_params: Optional[dict]=None, edit_params: Optional[dict]=None) -> dict:
    import sys, time
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.append(root_path)
    print('=' * 60)
    print('STARTING COMPLETE 3D EDITING PIPELINE')
    print('=' * 60)
    if not output_path.lower().endswith('.glb'):
        raise ValueError('output_path must end with .glb extension')
    results = {'input_model': input_model_path, 'mask_glb': mask_glb_path, 'render_dir': render_dir, 'final_output': output_path}
    if is_text:
        results['source_prompt'] = source_prompt
        results['target_prompt'] = target_prompt
    else:
        results['image_dir'] = image_dir
    render_results = run_3d_rendering(input_model_path, render_dir, **render_params or {})
    print('render is ok')
    results['rendering'] = render_results
    feature_results = run_feature_extraction(render_dir, **feature_params or {})
    results['features'] = feature_results
    print('feature is ok')
    mask_results = run_voxel_masking(mask_glb_path, render_dir, **mask_params or {})
    results['masking'] = mask_results
    print('mask is ok')
    start_time = time.perf_counter()
    edit_results = run_3d_editing(pipeline, render_dir, output_path, image_dir, is_text, source_prompt, target_prompt, **edit_params or {})
    results['editing'] = edit_results
    print('=' * 60)
    print('PIPELINE COMPLETED SUCCESSFULLY!')
    print('=' * 60)
    print(f'Final result: {output_path}')
    end_time = time.perf_counter()
    print('full run time:', end_time - start_time)
    return results

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--is_text', type=bool, default=True)
    parser.add_argument('--input_model', type=str, default='Experience/boy/model.glb')
    parser.add_argument('--mask_glb', type=str, default='vis/mask.glb')
    parser.add_argument('--render_dir', type=str, default='vis/render')
    parser.add_argument('--output_dir', type=str, default='vis/drag_output')
    parser.add_argument('--output_path', type=str, default='vis/drag_output/output.glb')
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--source_prompt', type=str, default='')
    parser.add_argument('--target_prompt', type=str, default='')
    parser.add_argument('--map_save_path1', type=str, default='lazydrag_maps_data.json')
    parser.add_argument('--map_save_path2', type=str, default='lazydrag_maps_data.json')
    parser.add_argument('--map_save_path3', type=str, default='lazydrag_maps_data.json')
    (parser.add_argument('--is_64', type=str, default='64'),)
    parser.add_argument('--use_logit_blend', action='store_true', help='')
    args = parser.parse_args()
    if args.is_text:
        pipeline = TrellisTextTo3DPipeline.from_pretrained(f'{DIR_PATH}/data/models/TRELLIS-text-large')
    pipeline.cuda()
    map_save_dir = os.path.dirname(args.map_save_path1)
    map_save_path = os.path.join(map_save_dir, 'lazydrag_maps_data_std.json')
    if args.is_64 == '64':
        map_path = map_to_std(raw_json_path=args.map_save_path1, out_json_path=map_save_path, ss_patch_size=4, slat_factor=2)
    elif args.is_64 == '16':
        map_path = map_to_std_16(raw_json_path=args.map_save_path1, out_json_path=map_save_path, ss_patch_size=4, slat_factor=2)
    else:
        map_path = map_to_std_16_with_32_and_64(raw_json_path1=args.map_save_path1, raw_json_path2=args.map_save_path2, raw_json_path3=args.map_save_path3, out_json_path=map_save_path)
    edit_params = {'map_path': map_path}
    if args.use_logit_blend:
        edit_params['use_logit_blend'] = True
    pipeline_result = run_complete_pipeline(pipeline=pipeline, input_model_path=args.input_model, mask_glb_path=args.mask_glb, render_dir=args.render_dir, output_path=args.output_path, image_dir=args.image_dir, is_text=args.is_text, source_prompt=args.source_prompt, target_prompt=args.target_prompt, edit_params=edit_params)
    glb_path = pipeline_result.get('editing')
    print(f'[PIPELINE_SUCCESS] {glb_path}', flush=True)
    sys.exit(0)
if __name__ == '__main__':
    main()