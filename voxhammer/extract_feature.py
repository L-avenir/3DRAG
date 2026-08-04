import os
import json
import torch
import utils3d
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from queue import Queue
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

torch.set_grad_enabled(False)


def voxelize_mesh(output_dir):
    mesh_path = os.path.join(output_dir, "mesh.ply")
    voxels_path = os.path.join(output_dir, "voxels.ply")
    if not os.path.exists(mesh_path):
        raise ValueError(f"Mesh file not found: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1 / 64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5)
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(
        vertices < 64
    ), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    utils3d.io.write_ply(voxels_path, vertices)
    print(f"Voxelized mesh saved to: {voxels_path}")
    return voxels_path


def get_data(frames, output_dir):
    with ThreadPoolExecutor(max_workers=16) as executor:

        def worker(view):
            image_path = os.path.join(output_dir, view["file_path"])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            c2w = torch.tensor(view["transform_matrix"])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view["camera_angle_x"]
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(
                torch.tensor(fov), torch.tensor(fov)
            )
            return {"image": image, "extrinsics": extrinsics, "intrinsics": intrinsics}

        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data


def extract_features(output_dir, model="dinov2_vitl14_reg", batch_size=16):
    dinov2_model = torch.hub.load("facebookresearch/dinov2", model, pretrained=True)
    dinov2_model.eval().cuda()
    transform = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    n_patch = 518 // 14
    transforms_path = os.path.join(output_dir, "transforms.json")
    mesh_path = os.path.join(output_dir, "mesh.ply")
    voxels_path = os.path.join(output_dir, "voxels.ply")
    if not os.path.exists(transforms_path):
        raise ValueError(f"Transforms file not found: {transforms_path}")
    if not os.path.exists(mesh_path):
        raise ValueError(f"Mesh file not found: {mesh_path}")
    if not os.path.exists(voxels_path):
        print("Voxelizing mesh...")
        voxelize_mesh(output_dir)
    ply_path = voxels_path
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, ThreadPoolExecutor(
            max_workers=8
        ) as saver_executor:

            def loader(dummy_param):
                try:
                    with open(transforms_path, "r") as f:
                        metadata = json.load(f)
                    frames = metadata["frames"]
                    data = []
                    for datum in get_data(frames, output_dir):
                        datum["image"] = transform(datum["image"])
                        data.append(datum)
                    positions = utils3d.io.read_ply(ply_path)[0]
                    load_queue.put((data, positions))
                except Exception as e:
                    print(f"Error loading data: {e}")

            loader_executor.map(loader, [None])

            def saver(pack, patchtokens, uv):
                try:
                    patchtokens = patchtokens.cpu()
                    uv = uv.cpu()
                    sampled = F.grid_sample(
                        patchtokens,
                        uv.unsqueeze(1),
                        mode="bilinear",
                        align_corners=False,
                    )
                    sampled = sampled.squeeze(2).permute(0, 2, 1).numpy()
                    pack["patchtokens"] = np.mean(sampled, axis=0).astype(np.float16)

                    save_path = os.path.join(output_dir, "features.npz")
                    np.savez_compressed(save_path, **pack)
                    print(f"[VERIFY] Checking file: {os.path.abspath(save_path)}")
                    print(f"[VERIFY] Exists: {os.path.exists(save_path)}")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise e

            data, positions = load_queue.get()
            positions = torch.from_numpy(positions).float().cuda()
            indices = ((positions + 0.5) * 64).long()
            assert torch.all(indices >= 0) and torch.all(
                indices < 64
            ), "Some vertices are out of bounds"
            n_views = len(data)
            N = positions.shape[0]
            pack = {"indices": indices.cpu().numpy().astype(np.uint8)}
            patchtokens_lst = []
            uv_lst = []
            for i in tqdm(
                range(0, n_views, batch_size), desc="Processing image batches"
            ):
                batch_data = data[i : i + batch_size]
                bs = len(batch_data)
                batch_images = torch.stack([d["image"] for d in batch_data]).cuda()
                batch_extrinsics = torch.stack(
                    [d["extrinsics"] for d in batch_data]
                ).cuda()
                batch_intrinsics = torch.stack(
                    [d["intrinsics"] for d in batch_data]
                ).cuda()
                features = dinov2_model(batch_images, is_training=True)
                uv = (
                    utils3d.torch.project_cv(
                        positions, batch_extrinsics, batch_intrinsics
                    )[0]
                    * 2
                    - 1
                )
                patchtokens = (
                    features["x_prenorm"][:, dinov2_model.num_register_tokens + 1 :]
                    .permute(0, 2, 1)
                    .reshape(bs, 1024, n_patch, n_patch)
                )
                patchtokens_lst.append(patchtokens)
                uv_lst.append(uv)
            patchtokens = torch.cat(patchtokens_lst, dim=0)
            uv = torch.cat(uv_lst, dim=0)
            saver_executor.submit(saver, pack, patchtokens, uv)
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")


if __name__ == "__main__":
    render_dir = "/render/path"
    extract_features(render_dir, model="dinov2_vitl14_reg", batch_size=10)
