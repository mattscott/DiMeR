import importlib
import sys

# List modules to force reload
modules_to_reload = ['pipeline.mv_kiss3d_wrapper']

for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

import os
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
parser = argparse.ArgumentParser(description='DiMeR Bundle Image-to-3D Demo')
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

@spaces.GPU(duration=120)
def bundle_image2mesh_(bundle_image, strength1=0.5, strength2=0.95, enable_redux=True):
    global mesh_cache 
    print (f"bundle_image2mesh_() called")

    allocated_vram, reserved_vram, total_vram = get_vram_usage()
    print(f"#0 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")
    allocated_vram, reserved_vram, total_vram = get_vram_usage(device_id=1)
    print(f"#1 VRAM Usage: Allocated={allocated_vram:.2f}GB, Reserved={reserved_vram:.2f}GB, Total={total_vram:.2f}GB")

    # Convert bundle image to PIL Image for input
    input_pil = Image.fromarray(bundle_image)
    
    # Convert bundle image to tensor and split into individual views for reference
    bundle_tensor = torch.tensor(bundle_image).permute(2,0,1)/255  # Convert to C,H,W format
    
    # Split into RGB and normal views
    rgb_views = []
    normal_views = []
    for i in range(4):  # 4 views
        # Extract RGB view (top row)
        rgb_view = bundle_tensor[:, :512, i*512:(i+1)*512]
        rgb_views.append(rgb_view)
        
        # Extract normal view (bottom row)
        normal_view = bundle_tensor[:, 512:, i*512:(i+1)*512]
        normal_views.append(normal_view)
    
    # Stack views into tensors
    rgb_tensor = torch.stack(rgb_views)  # Shape: [4, C, H, W]
    normal_tensor = torch.stack(normal_views)  # Shape: [4, C, H, W]
    
    # Generate 3D model
    gen_save_path, recon_mesh_path = image2mesh_main(
        k3d_wrapper, 
        input_pil,  # Pass as PIL Image for input
        bundle_tensor,  # Pass full bundle tensor as reference
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
            # DiMeR: Bundle Image-to-3D Generation""")
            
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
        with gr.TabItem('Bundle Image-to-3D', id='tab_bundle_image_to_3d'):
            gr.Markdown("Upload a processed bundle image (containing RGB and normal maps) and click 'Generate 3D Model' to create a 3D mesh.")
            with gr.Row():
                with gr.Column(scale=1):
                    bundle_image = gr.Image(label="Bundle Image", type="numpy", interactive=True)
                    
                    with gr.Accordion("Advanced Parameters", open=False):
                        strength1 = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Strength 1")
                        strength2 = gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.05, label="Strength 2")
                        enable_redux = gr.Checkbox(value=True, label="Enable Redux")
                    
                    btn_generate = gr.Button("Generate 3D Model", elem_classes=["orange-button"])

                with gr.Column(scale=1):
                    output_image = gr.Image(label="Processed Image", interactive=False, width=800, height=350)
                    output_mesh = gr.Model3D(label="3D Mesh Viewer", interactive=False, height=300)
                    download_btn = gr.DownloadButton(label="Download Mesh", interactive=False)

    # Button Click Events
    btn_generate.click(
        fn=bundle_image2mesh_,
        inputs=[bundle_image, strength1, strength2, enable_redux],
        outputs=[output_image, output_mesh, download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn]
    )

    with gr.Row():
        gr.Markdown(_CITE_)

# Modify launch parameters to ensure background processing can continue
demo.launch() 