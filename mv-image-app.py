import importlib
import sys

# List modules to force reload
modules_to_reload = ['pipeline.mv_kiss3d_wrapper']

for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

sys.path.insert(0, 'custom_diffusers')

import gradio as gr
import subprocess
import spaces
import ctypes
import shlex
import torch
import argparse
import torchvision
import numpy as np
import gc

from PIL import Image
print(f'gradio version: {gr.__version__}')

# Add command line argument parsing
parser = argparse.ArgumentParser(description='DiMeR Multi-View Image-to-3D Demo')
parser.add_argument('--ui_only', action='store_true', help='Only load the UI interface, do not initialize models (for UI debugging)')
args = parser.parse_args()

UI_ONLY_MODE = args.ui_only
print(f"UI_ONLY_MODE: {UI_ONLY_MODE}")

# Status variables for tracking if image has been processed
processed_image = False

@spaces.GPU
def check_gpu():
    gc.collect()
    torch.cuda.empty_cache()

    print(f"torch.cuda.is_available:{torch.cuda.is_available()}")
    print("Device count:", torch.cuda.device_count()) 

    # test nvdiffrast
    import nvdiffrast.torch as dr
    dr.RasterizeCudaContext(device="cuda:1")
    print("nvdiffrast initialized successfully")       

# Only check GPU in non-UI debug mode
if not UI_ONLY_MODE:
    check_gpu()

import base64
import re
import sys
import shutil
import json
import requests
import threading
import time
import trimesh
import random
import time
import numpy as np

# Only import video rendering module and initialize models in non-UI debug mode
if not UI_ONLY_MODE:
    from pipeline.mv_kiss3d_wrapper import init_wrapper_from_config, image2mesh_main
    from pipeline.utils import TMP_DIR, OUT_DIR

# Add logo file path and hyperlinks
LOGO_PATH = "app_assets/logo_temp_.png"
ARXIV_LINK = "https://arxiv.org/pdf/2504.17670"
GITHUB_LINK = "https://github.com/lutao2021/DiMeR"

# Only initialize models in non-UI debug mode
if not UI_ONLY_MODE:
    k3d_wrapper = init_wrapper_from_config('./pipeline/pipeline_config/default.yaml')
    from models.ISOMER.scripts.utils import fix_vert_color_glb
    torch.backends.cuda.matmul.allow_tf32 = True

TEMP_MESH_ADDRESS=''
mesh_cache = None
preprocessed_input_image = None

def get_vram_usage(device_id=0):
    """
    Measures VRAM usage on a specified CUDA device.

    Args:
        device_id (int, optional): The ID of the CUDA device. Defaults to 0.

    Returns:
        tuple: A tuple containing allocated VRAM, reserved VRAM, and total VRAM, all in GB.
    """
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available.")

    allocated_vram = torch.cuda.memory_allocated(device_id) / (1024**3)  # Convert to GB
    reserved_vram = torch.cuda.memory_reserved(device_id) / (1024**3)  # Convert to GB
    total_vram = torch.cuda.get_device_properties(device_id).total_memory / (1024**3) # Convert to GB

    return allocated_vram, reserved_vram, total_vram

def save_cached_mesh():
    global mesh_cache
    print('save_cached_mesh() called')
    return mesh_cache

def combine_multi_view_images(front_img, back_img, left_img, right_img):
    """Combine 4 view images into a single bundle image"""
    # Convert PIL images to tensors
    front_tensor = torchvision.transforms.functional.to_tensor(front_img)
    back_tensor = torchvision.transforms.functional.to_tensor(back_img)
    left_tensor = torchvision.transforms.functional.to_tensor(left_img)
    right_tensor = torchvision.transforms.functional.to_tensor(right_img)
    
    # Stack images in the correct order (front, right, back, left)
    rgb_images = torch.stack([front_tensor, right_tensor, back_tensor, left_tensor])
    
    # Create normal maps (placeholder for now)
    normal_images = torch.ones_like(rgb_images)
    
    # Stack RGB and normal images vertically
    bundle_image = torch.cat([rgb_images, normal_images], dim=0)
    
    # Create a grid with 4 images per row, but stack the normal images below
    rgb_grid = torchvision.utils.make_grid(rgb_images, nrow=4, padding=0)
    normal_grid = torchvision.utils.make_grid(normal_images, nrow=4, padding=0)
    bundle_image = torch.cat([rgb_grid, normal_grid], dim=1)  # Stack vertically
    
    # Save intermediate result
    save_path = os.path.join(TMP_DIR, f'{k3d_wrapper.uuid}_ref_3d_bundle_image.png')
    torchvision.utils.save_image(bundle_image, save_path)
    
    return bundle_image, save_path

@spaces.GPU(duration=120)
def mv_image2mesh_preprocess_(front_img, back_img, left_img, right_img, seed):
    global preprocessed_input_image

    print (f"mv_image2mesh_preprocess_() called with seed: {seed}")
    
    allocated_vram, reserved_vram, total_vram = get_vram_usage()
    print(f"#0 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")
    allocated_vram, reserved_vram, total_vram = get_vram_usage(device_id=1)
    print(f"#1 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")

    seed = int(seed) if seed is not None else None
    
    # Combine the multi-view images into a bundle
    reference_3d_bundle_image, reference_save_path = combine_multi_view_images(front_img, back_img, left_img, right_img)
    preprocessed_input_image = front_img

    allocated_vram, reserved_vram, total_vram = get_vram_usage()
    print(f"#0 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")
    allocated_vram, reserved_vram, total_vram = get_vram_usage(device_id=1)
    print(f"#1 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")

    return reference_save_path

@spaces.GPU(duration=120)
def mv_image2mesh_main_(reference_3d_bundle_image, strength1=0.5, strength2=0.95, enable_redux=True):
    global mesh_cache 
    print (f"mv_image2mesh_main_() called")

    allocated_vram, reserved_vram, total_vram = get_vram_usage()
    print(f"#0 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")
    allocated_vram, reserved_vram, total_vram = get_vram_usage(device_id=1)
    print(f"#1 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")

    # Convert reference image to tensor
    reference_3d_bundle_image = torch.tensor(reference_3d_bundle_image).permute(2,0,1)/255

    # Generate 3D model
    gen_save_path, recon_mesh_path = image2mesh_main(
        k3d_wrapper, 
        preprocessed_input_image,  # Use the preprocessed input image
        reference_3d_bundle_image, 
        strength1=strength1, 
        strength2=strength2, 
        enable_redux=enable_redux
    )
    
    mesh_cache = recon_mesh_path

    allocated_vram, reserved_vram, total_vram = get_vram_usage()
    print(f"#0 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")
    allocated_vram, reserved_vram, total_vram = get_vram_usage(device_id=1)
    print(f"#1 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")

    return gen_save_path, recon_mesh_path, mesh_cache

if not UI_ONLY_MODE:
    torch.set_grad_enabled(False)

with gr.Blocks(css="""
    .orange-button {
        background-color: #FF8C00 !important;
        border-color: #FF8C00 !important;
        color: black !important;
    }
    .gradio-container {
        max-width: 1000px;
        margin: auto;
        width: 100%;
    }
    #center-align-column {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    #right-align-column {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    h1 {text-align: center;}
    h2 {text-align: center;}
    h3 {text-align: center;}
    p {text-align: center;}
    img {text-align: right;}
    .right {
        display: block;
        margin-left: auto;
    }
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    #content-container {
        max-width: 1200px;
        margin: 0 auto;
    }
""", elem_id="col-container") as demo:
    with gr.Row(elem_id="content-container"):
        with gr.Column(scale=7, elem_id="center-align-column"):
            gr.Markdown(f"""
            # Official 🤗 Gradio Demo
            # DiMeR: Multi-View Image-to-3D Generation""")
            
            gr.HTML(f"""
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
                <a href="{ARXIV_LINK}" target="_blank">
                    <img src="https://img.shields.io/badge/arXiv-Link-red" alt="arXiv">
                </a>
                <a href="{GITHUB_LINK}" target="_blank">
                    <img src="https://img.shields.io/badge/GitHub-Repo-blue" alt="GitHub">
                </a>
            </div>
            """)

    _STAR_ = f"""
    <h2>If DiMeR is helpful, please help to ⭐ the <a href={GITHUB_LINK} target='_blank'>Github Repo</a>. Sincerely Thanks!</h2>
    """

    _CITE_ = r"""

    <h2>📝 Citation</h2>

    <h2>If you find our work useful for your research or applications, please cite using the following papers:</h2>

    ```bibtex
    @article{jiang2025dimer,
    title={DiMeR: Disentangled Mesh Reconstruction Model},
    author={Jiang, Lutao and Lin, Jiantao and Chen, Kanghao and Ge, Wenhang and Yang, Xin and Jiang, Yifan and Lyu, Yuanhuiyi and Zheng, Xu and Chen, Yingcong},
    journal={arXiv preprint arXiv:2504.17670},
    year={2025}
    }

    @article{lin2025kiss3dgenrepurposingimagediffusion,
    title={Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation},
    author={Jiantao Lin, Xin Yang, Meixi Chen, Yingjie Xu, Dongyu Yan, Leyi Wu, Xinli Xu, Lie XU, Shunsi Zhang, Ying-Cong Chen},
    journal={arXiv preprint arXiv:2503.01370},
    year={2025}
    }

    ```

    📋 **License**

    Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/InstantMesh/blob/main/LICENSE) for details.

    📧 **Contact**

    If you have any questions, feel free to open a discussion or contact us at <b>ljiang553@connect.hkust-gz.edu.cn</b>.
    """

    gr.Markdown(_STAR_)

    with gr.Tabs() as main_tabs:
        with gr.TabItem('Multi-View Image-to-3D', id='tab_mv_image_to_3d'):
            gr.Markdown("Upload front, back, left, and right view images and click 'Generate 3D Model' to create a 3D mesh.")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        front_img = gr.Image(label="Front View", type="pil", interactive=True)
                        back_img = gr.Image(label="Back View", type="pil", interactive=True)
                    with gr.Row():
                        left_img = gr.Image(label="Left View", type="pil", interactive=True)
                        right_img = gr.Image(label="Right View", type="pil", interactive=True)
                    
                    with gr.Accordion("Advanced Parameters", open=False):
                        seed = gr.Number(value=4242, label="Seed")
                        strength1 = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Strength 1")
                        strength2 = gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.05, label="Strength 2")
                        enable_redux = gr.Checkbox(value=True, label="Enable Redux")
                        use_controlnet = gr.Checkbox(value=True, label="Use ControlNet")
                        camera_radius = gr.Slider(minimum=3.0, maximum=6.0, value=3.5, step=0.01, label="Camera Radius")
                    
                    btn_generate = gr.Button("Generate 3D Model", elem_classes=["orange-button"])

                with gr.Column(scale=1):
                    output_image = gr.Image(label="Processed Image", interactive=False, width=800, height=350)
                    output_mesh = gr.Model3D(label="3D Mesh Viewer", interactive=False, height=300)
                    download_btn = gr.DownloadButton(label="Download Mesh", interactive=False)

    # Button Click Events
    btn_generate.click(
        fn=mv_image2mesh_preprocess_,
        inputs=[front_img, back_img, left_img, right_img, seed],
        outputs=[output_image]
    ).then(
        fn=mv_image2mesh_main_,
        inputs=[output_image, strength1, strength2, enable_redux],
        outputs=[output_image, output_mesh, download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn]
    )

    with gr.Row():
        gr.Markdown(_CITE_)

# Modify launch parameters to ensure background processing can continue
demo.launch() 