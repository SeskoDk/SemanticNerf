# SemanticNeRF

SemanticNeRF is a pipeline for **3D semantic reconstruction of wheat
scenes** using NeRF.\
The system combines **COLMAP**, **object detection**, **segmentation**,
and **NeRF-based volumetric reconstruction** to produce a **3D semantic
volume** that can be used for downstream tasks such as biomass
estimation.

------------------------------------------------------------------------

## Installation

### 1. Create a Python environment

    python -m venv .venv

Activate the environment.

Windows:

    .venv\Scripts\activate

Linux / Mac:

    source .venv/bin/activate

------------------------------------------------------------------------

### 2. Install PyTorch

Example for CUDA 12.6:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

------------------------------------------------------------------------

### 3. Install TinyCudaNN

    pip install ninja cmake
    pip install --upgrade pip wheel
    pip install "setuptools<70"
    pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

------------------------------------------------------------------------

## 4. Download Required Tools

### SAM Model

Download a pretrained **SAM model**:

https://docs.ultralytics.com/models/sam-2/#interactive-segmentation

Store it under

    tools/pretrained_models/sam/

### COLMAP

Download COLMAP:

https://github.com/colmap/colmap/releases

Example (Windows):

    colmap-x64-windows-cuda.zip

------------------------------------------------------------------------

## Data 

The RGB images used for reconstruction must be placed in the following directory structure:

The RGB images must be stored in the following structure:
```
data/
└── raw/
    └── <scene_name>/
        ├── image_0001.jpg
        ├── image_0002.jpg
        ├── image_0003.jpg
        └── ...
```

## 3D Segmentation Pipeline

Replace `<scene_name>` with your dataset configuration.

------------------------------------------------------------------------

### 1. Run COLMAP

    python -m src.colmap.run_colmap --config_path cfg/<scene_name>.yml

------------------------------------------------------------------------

### 2. Normalize the Scene

Creates

    transforms_<scene_name>.json

Location:

    results/<scene_name>/

Command:

    python -m src.colmap.normalize_scene --config_path cfg/<scene_name>.yml

------------------------------------------------------------------------

### 3. Detect Wheat Heads

Output:

    results/<scene_name>/yolo_bbx.json

Command:

    python -m src.detection.bbx_detection --config_path cfg/<scene_name>.yml

------------------------------------------------------------------------

### 4. Generate Segmentation Masks

Output:

    results/<scene_name>/segmentation_mask/

Command:

    python -m src.segmentation.bbox_to_mask --config_path cfg/<scene_name>.yml

------------------------------------------------------------------------

### 5. Precompute Rays

    python -m src.rays.generation --config_path cfg/<scene_name>.yml --display_rays --only_inside --max_results 100

------------------------------------------------------------------------

### 6. Train the Semantic NeRF

    python -m src.train --config_path cfg/<scene_name>.yml

Outputs

Model:

    results/<scene_name>/checkpoints/sem_nerf_model.pth

Renderings:

    results/<scene_name>/novel_views/

------------------------------------------------------------------------

### 7. Create 3D Volume

Output:

    results/<scene_name>/volume/volume.npz

Command:

    python -m src.create_volume --config_path cfg/<scene_name>.yml

------------------------------------------------------------------------

### 8. Render Orthographic Projection

Output:

    results/<scene_name>/axis_renders/

Command:

    python -m src.render.render --config_path cfg/<scene_name>.yml

------------------------------------------------------------------------

### 9. Create 2D Mask Using SAM

Procedure:

Click **four points inside the metal frame** and press **Enter**.

Output:

    results/<scene_name>/axis_renders/topdown_filled_mask.png

Command:

    python -m src.render_mask_with_sam --config_path cfg/<scene_name>.yml

------------------------------------------------------------------------

### 10. Create Semantic Volume

Outputs:

    results/<scene_name>/volume/volume_semantic.npz
    results/<scene_name>/volume/volume_semantic_red.npz

Command:

    python -m src.postprocess --config_path cfg/<scene_name>.yml

------------------------------------------------------------------------

### 11. Compute Biomass

    python -m src.biomass

------------------------------------------------------------------------

## Output Overview

    results/<scene_name>/

    checkpoints/
        sem_nerf_model.pth

    axis_renders/
        z_rgb.png
        topdown_filled_mask.png

    volume/
        volume.npz
        volume_semantic.npz
        volume_semantic_red.npz

    novel_views/
        rgb
        depth
        semantic
