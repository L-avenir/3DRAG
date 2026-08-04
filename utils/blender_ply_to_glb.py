import bpy
import sys
import os

def ensure_uv(obj, uv_name='UVMap'):
    me = obj.data
    if me.uv_layers.get(uv_name) is None:
        me.uv_layers.new(name=uv_name)
    uv_layer = me.uv_layers[uv_name]
    for loop in me.loops:
        uv_layer.data[loop.index].uv = (0.5, 0.5)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_ply(ply_path: str):
    if hasattr(bpy.ops.wm, 'ply_import'):
        bpy.ops.wm.ply_import(filepath=ply_path)
    else:
        try:
            bpy.ops.preferences.addon_enable(module='io_mesh_ply')
        except Exception:
            pass
        if hasattr(bpy.ops.import_mesh, 'ply'):
            bpy.ops.import_mesh.ply(filepath=ply_path)
        else:
            raise RuntimeError('No PLY import operator found (wm.ply_import / import_mesh.ply).')
    obj = bpy.context.selected_objects[0]
    return obj

def export_glb(glb_path: str, obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB', use_selection=True, export_normals=True, export_texcoords=True, export_materials='NONE')

def main():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else []
    if len(argv) != 2:
        raise SystemExit('Usage: blender -b -P blender_ply_to_glb.py -- in.ply out.glb')
    (in_ply, out_glb) = argv
    clear_scene()
    obj = import_ply(in_ply)
    obj.name = 'Sphere'
    obj.data.name = 'Sphere'
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    ensure_uv(obj)
    export_glb(out_glb, obj)
    print(f'[OK] wrote: {out_glb}')
if __name__ == '__main__':
    main()
'blender -b -P utils/blender_ply_to_glb.py -- process/compare_test/child/mask_tmp.ply process/compare_test/child/mask.glb'