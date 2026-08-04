import os
import bpy
import json
import math
import numpy as np
from mathutils import Vector
from typing import Dict, Tuple

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def radical_inverse(base, n):
        val = 0
        inv_base = 1.0 / base
        inv_base_n = inv_base
        while n > 0:
            digit = n % base
            val += digit * inv_base_n
            n //= base
            inv_base_n *= inv_base
        return val

    def halton_sequence(dim, n):
        return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

    def hammersley_sequence(dim, n, num_samples):
        return [n / num_samples] + halton_sequence(dim - 1, n)
    (u, v) = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

class BpyRenderer:

    def __init__(self, resolution: int=512, engine: str='CYCLES', geo_mode: bool=False, split_normal: bool=False):
        self.resolution = resolution
        self.engine = engine
        self.geo_mode = geo_mode
        self.split_normal = split_normal
        self.import_functions = self._setup_import_functions()

    def _setup_import_functions(self):
        import_functions = {'obj': bpy.ops.wm.obj_import, 'glb': bpy.ops.import_scene.gltf, 'gltf': bpy.ops.import_scene.gltf, 'usd': bpy.ops.import_scene.usd, 'fbx': bpy.ops.import_scene.fbx, 'stl': bpy.ops.import_mesh.stl, 'usda': bpy.ops.import_scene.usda, 'dae': bpy.ops.wm.collada_import, 'ply': bpy.ops.wm.ply_import, 'abc': bpy.ops.wm.alembic_import, 'blend': bpy.ops.wm.append}
        return import_functions

    def init_render_settings(self):
        bpy.context.scene.render.engine = self.engine
        bpy.context.scene.render.resolution_x = self.resolution
        bpy.context.scene.render.resolution_y = self.resolution
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.view_settings.view_transform = 'Standard'
        if self.engine == 'CYCLES':
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.scene.cycles.samples = 128 if not self.geo_mode else 1
            bpy.context.scene.cycles.filter_type = 'BOX'
            bpy.context.scene.cycles.filter_width = 1
            bpy.context.scene.cycles.diffuse_bounces = 1
            bpy.context.scene.cycles.glossy_bounces = 1
            bpy.context.scene.cycles.transparent_max_bounces = 3 if not self.geo_mode else 0
            bpy.context.scene.cycles.transmission_bounces = 3 if not self.geo_mode else 1
            bpy.context.scene.cycles.use_denoising = True
            try:
                bpy.context.scene.cycles.device = 'GPU'
                bpy.context.preferences.addons['cycles'].preferences.get_devices()
                bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            except:
                pass

    def init_scene(self):
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    def init_camera(self):
        cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_height = cam.data.sensor_width = 32
        cam_constraint = cam.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        cam_empty = bpy.data.objects.new('Empty', None)
        cam_empty.location = (0, 0, 0)
        bpy.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty
        return cam

    def init_lighting(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()
        default_light = bpy.data.objects.new('Default_Light', bpy.data.lights.new('Default_Light', type='POINT'))
        bpy.context.collection.objects.link(default_light)
        default_light.data.energy = 1000
        default_light.location = (4, 1, 6)
        default_light.rotation_euler = (0, 0, 0)
        top_light = bpy.data.objects.new('Top_Light', bpy.data.lights.new('Top_Light', type='AREA'))
        bpy.context.collection.objects.link(top_light)
        top_light.data.energy = 10000
        top_light.location = (0, 0, 10)
        top_light.scale = (100, 100, 100)
        bottom_light = bpy.data.objects.new('Bottom_Light', bpy.data.lights.new('Bottom_Light', type='AREA'))
        bpy.context.collection.objects.link(bottom_light)
        bottom_light.data.energy = 1000
        bottom_light.location = (0, 0, -10)
        bottom_light.rotation_euler = (0, 0, 0)
        return {'default_light': default_light, 'top_light': top_light, 'bottom_light': bottom_light}

    def load_object(self, object_path: str):
        file_extension = object_path.split('.')[-1].lower()
        if file_extension not in self.import_functions:
            raise ValueError(f'Unsupported file type: {file_extension}')
        import_function = self.import_functions[file_extension]
        print(f'Loading object from {object_path}')
        if file_extension == 'blend':
            import_function(directory=object_path, link=False)
        elif file_extension in {'glb', 'gltf'}:
            import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
        else:
            import_function(filepath=object_path)

    def make_materials_shadeless(self):
        for mat in bpy.data.materials:
            if not mat.use_nodes or not mat.node_tree:
                continue
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            principled_node = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if output_node and principled_node:
                emission_node = nodes.new(type='ShaderNodeEmission')
                emission_node.inputs['Strength'].default_value = 1.0
                base_color_input = principled_node.inputs.get('Base Color')
                if base_color_input and base_color_input.is_linked:
                    texture_socket = base_color_input.links[0].from_socket
                    links.new(texture_socket, emission_node.inputs['Color'])
                elif base_color_input:
                    emission_node.inputs['Color'].default_value = base_color_input.default_value
                links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    def delete_invisible_objects(self):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.scene.objects:
            if obj.hide_viewport or obj.hide_render:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.hide_select = False
                obj.select_set(True)
        bpy.ops.object.delete()
        invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
        for col in invisible_collections:
            bpy.data.collections.remove(col)

    def unhide_all_objects(self):
        for obj in bpy.context.scene.objects:
            obj.hide_set(False)

    def convert_to_meshes(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'][0]
        for obj in bpy.context.scene.objects:
            obj.select_set(True)
        bpy.ops.object.convert(target='MESH')

    def triangulate_meshes(self):
        bpy.ops.object.select_all(action='DESELECT')
        objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.reveal()
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')

    def split_mesh_normal(self):
        bpy.ops.object.select_all(action='DESELECT')
        objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.split_normals()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')

    def delete_custom_normals(self):
        for this_obj in bpy.data.objects:
            if this_obj.type == 'MESH':
                bpy.context.view_layer.objects.active = this_obj
                bpy.ops.mesh.customdata_custom_splitnormals_clear()

    def override_material(self):
        new_mat = bpy.data.materials.new(name='Override0123456789')
        new_mat.use_nodes = True
        new_mat.node_tree.nodes.clear()
        bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
        bsdf.inputs[1].default_value = 1
        output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
        new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        bpy.context.scene.view_layers['View Layer'].material_override = new_mat

    def scene_bbox(self) -> Tuple[Vector, Vector]:
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
        for obj in scene_meshes:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple((min(x, y) for (x, y) in zip(bbox_min, coord)))
                bbox_max = tuple((max(x, y) for (x, y) in zip(bbox_max, coord)))
        if not found:
            raise RuntimeError('no objects in scene to compute bounding box for')
        return (Vector(bbox_min), Vector(bbox_max))

    def normalize_scene(self) -> Tuple[float, Vector]:
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new('ParentEmpty', None)
            bpy.context.scene.collection.objects.link(scene)
            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]
        (bbox_min, bbox_max) = self.scene_bbox()
        print(f'[INFO] Bounding box: {bbox_min}, {bbox_max}')
        scale = 1 / max(bbox_max - bbox_min)
        scene.scale = scene.scale * scale
        bpy.context.view_layer.update()
        (bbox_min, bbox_max) = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action='DESELECT')
        return (scale, offset)

    def get_transform_matrix(self, obj: bpy.types.Object) -> list:
        (pos, rt, _) = obj.matrix_world.decompose()
        rt = rt.to_matrix()
        matrix = []
        for ii in range(3):
            a = []
            for jj in range(3):
                a.append(rt[ii][jj])
            a.append(pos[ii])
            matrix.append(a)
        matrix.append([0, 0, 0, 1])
        return matrix

    def render_object(self, file_path: str, output_dir: str, num_views: int=150, scale: float=1.0, offset: Vector=None, save_mesh: bool=True) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        self.init_render_settings()
        if file_path.endswith('.blend'):
            self.delete_invisible_objects()
        else:
            self.init_scene()
            self.load_object(file_path)
            if not self.geo_mode:
                self.make_materials_shadeless()
                print('[INFO] Materials converted to shadeless for pure albedo.')
            if self.split_normal:
                self.split_mesh_normal()
        print('[INFO] Scene initialized.')
        if offset is None:
            (scale, offset) = self.normalize_scene()
            print(f'[INFO] Scene normalized with auto scale: {scale}, offset: {offset}')
        else:
            scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
            if len(scene_root_objects) > 1:
                scene = bpy.data.objects.new('ParentEmpty', None)
                bpy.context.scene.collection.objects.link(scene)
                for obj in scene_root_objects:
                    obj.parent = scene
            else:
                scene = scene_root_objects[0]
            scene.scale = scene.scale * scale
            bpy.context.view_layer.update()
            scene.matrix_world.translation += offset
            bpy.ops.object.select_all(action='DESELECT')
            print(f'[INFO] Scene scaled with specified scale: {scale}, offset: {offset}')
        cam = self.init_camera()
        self.init_lighting()
        print('[INFO] Camera and lighting initialized.')
        if self.geo_mode:
            self.override_material()
        yaws = []
        pitchs = []
        offset_random = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            (y, p) = sphere_hammersley_sequence(i, num_views, offset_random)
            yaws.append(y)
            pitchs.append(p)
        radius = [2] * num_views
        fov = [40 / 180 * np.pi] * num_views
        views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for (y, p, r, f) in zip(yaws, pitchs, radius, fov)]
        to_export = {'aabb': [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], 'scale': scale, 'offset': [offset.x, offset.y, offset.z], 'frames': []}
        for (i, view) in enumerate(views):
            cam.location = (view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']), view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']), view['radius'] * np.sin(view['pitch']))
            cam.data.lens = 16 / np.tan(view['fov'] / 2)
            bpy.context.scene.render.filepath = os.path.join(output_dir, f'{i:03d}.png')
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            metadata = {'file_path': f'{i:03d}.png', 'camera_angle_x': view['fov'], 'transform_matrix': self.get_transform_matrix(cam)}
            to_export['frames'].append(metadata)
        with open(os.path.join(output_dir, 'transforms.json'), 'w') as f:
            json.dump(to_export, f, indent=4)
        mesh_file_path = None
        if save_mesh:
            try:
                self.unhide_all_objects()
                self.convert_to_meshes()
                self.triangulate_meshes()
                print('[INFO] Meshes triangulated.')
                ply_path = os.path.join(output_dir, 'mesh.ply')
                try:
                    bpy.ops.wm.ply_export(filepath=ply_path)
                    mesh_file_path = ply_path
                    print('[INFO] Mesh file saved.')
                except AttributeError:
                    try:
                        bpy.ops.export_mesh.ply(filepath=ply_path)
                        mesh_file_path = ply_path
                        print('[INFO] Mesh file saved.')
                    except AttributeError:
                        print('[WARNING] PLY export not available, skipping mesh export')
            except Exception as e:
                print(f'[WARNING] Mesh export failed: {e}')
        return {'rendered': True, 'num_views': num_views, 'output_dir': output_dir, 'transforms_file': os.path.join(output_dir, 'transforms.json'), 'mesh_file': mesh_file_path}

def render_3d_model(file_path: str, output_dir: str, num_views: int=150, scale: float=1.0, offset: Vector=None, resolution: int=512, engine: str='CYCLES', geo_mode: bool=False, split_normal: bool=False, save_mesh: bool=True) -> Dict:
    renderer = BpyRenderer(resolution=resolution, engine=engine, geo_mode=geo_mode, split_normal=split_normal)
    return renderer.render_object(file_path, output_dir, num_views, scale, offset, save_mesh)
if __name__ == '__main__':
    file_path = 'output.glb'
    render_dir = 'shadeless'
    render_3d_model(file_path=file_path, output_dir=render_dir)