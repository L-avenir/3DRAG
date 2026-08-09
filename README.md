# 3DRAG: 3D-Native Object Drag Editing with Generative Models

This repository is the official open-source implementation of our paper accepted by ACM Multimedia 2026 (ACM MM 2026).

The original 3DRAG code is released under the [MIT License](LICENSE). Third-party code, models, and dependencies are subject to their respective licenses. Because some third-party components impose non-commercial research restrictions, the complete runtime environment may be suitable only for non-commercial research use.

## 3DRAG-Bench

**3DRAG-Bench**, the dataset introduced in our paper, is now publicly available on Hugging Face: [AeTherRaIn/3DRAG-Bench](https://huggingface.co/datasets/AeTherRaIn/3DRAG-Bench).

## Installation

Our environment setup is built upon [TRELLIS](https://github.com/Microsoft/TRELLIS). Follow the steps below to configure the dependencies:

### Installation Steps
1. Create and activate the environment with the requirements from `environment.yml`
```bash
conda env create -f environment.yml
``` 

2. Install requirements from GitHub projects
```bash
pip install git+https://github.com/huanngzh/bpy-renderer.git
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
```

## Model Zoo & Weight Placement

### Step 1 : Directory Structure
Ensure your `data/models/directory` is organized as follows to match the internal path resolution:
```text
your-project-dir/
└── data/
    └── models/
        ├── clip-vit-large-patch14
        ├── dinov2
        ├── TRELLIS-image-large
        ├── TRELLIS-text-large
        └── dinov2_vitl14_reg.pth
```

### Step 2: Pipeline Configuration
You must manually configure the structural links between the encoders and decoders in your pipeline.json files.

#### Text-Conditioned Pipeline
Download both **TRELLIS-image-large** and **TRELLIS-text-large** model, and edit `TRELLIS-text-large/pipeline.json` :
```json
{
    "sparse_structure_encoder": "path/to/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16",
    "sparse_structure_decoder": "path/to/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
    "slat_encoder": "path/to/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16",
    "slat_decoder_gs": "path/to/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
    "slat_decoder_rf": "path/to/TRELLIS-image-large/ckpts/slat_dec_rf_swin8_B_64l8r16_fp16",
    "slat_decoder_mesh": "path/to/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16",
}
```

## Usage Guide
### 1. Data Preparation
Your input assets should follow the standardized data format:

```text
your-model-name/
├── model.glb
└── dataset_input_clean.json
```

### 2. Running the Inference UI
Launch the interactive visualization and editing tool:
```bash
python vis_main.py
```

After the Gradio server initializes, open the provided URL and input the absolute path of your model directory.

### 3. Interactive Editing Flow
**Initialization (Phase 1)**: Click the First Button. 

The system generates the Voxelized Representation and the Simplified Vector Field derived from the generative prior. 

**Optimization (Phase 2)**: Click the Second Button. 

The system performs the drag-editing operation. 

And the result will be exported as output.glb in the output/ directory within your input path.
