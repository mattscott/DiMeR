import os
import uuid
import torch
import torchvision
import numpy as np
import yaml
import logging
import random
import gc

from PIL import Image
from einops import rearrange
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict, Union, Any
from omegaconf import OmegaConf
from models.DiMeR.utils.train_util import instantiate_from_config
from huggingface_hub import hf_hub_download

from pipeline.utils import logger, TMP_DIR, OUT_DIR
from pipeline.utils import lrm_reconstruct, isomer_reconstruct, preprocess_input_image, DiMeR_reconstruct
from utils.tool import get_background

#from pipeline.custom_pipelines import FluxPriorReduxPipeline, FluxControlNetImg2ImgPipeline, FluxImg2ImgPipeline, FluxNormalPipeline
from diffusers import FluxPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler, FluxTransformer2DModel, AutoencoderTiny
from diffusers.models.controlnets.controlnet_flux import FluxMultiControlNetModel, FluxControlNetModel
from diffusers.schedulers import FlowMatchHeunDiscreteScheduler

logger = logging.getLogger(__name__)
access_token = os.getenv("HUGGINGFACE_TOKEN")

class ModelLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        self.models = {}
        self.dtype_ = {
            'fp8': torch.float8_e4m3fn,
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32
        }

    def load_normals_model(self):

        logger.info('==> Loading Normals model ...')

        # Create predictor instance
        normals_pipe = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1')

        self.models['normals'] = normals_pipe

        return self.models['normals']

    def load_reconstruction_model(self):
        if 'reconstruction' not in self.models:
            logger.info('==> Loading reconstruction model ...')
            recon_device = self.config['reconstruction'].get('device', 'cuda:1')
            recon_model_config = OmegaConf.load(self.config['reconstruction']['model_config'])
            recon_model = instantiate_from_config(recon_model_config.model_config)
            model_ckpt_path = hf_hub_download(repo_id="LutaoJiang/DiMeR", filename="DiMeR_geometry.ckpt", repo_type="model")
            state_dict = torch.load(model_ckpt_path, map_location='cuda:1')
            state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
            recon_model.load_state_dict(state_dict, strict=True)
            recon_model.to(recon_device)
            recon_model.eval()
            
            self.models['reconstruction'] = recon_model
        return self.models['reconstruction']

    def load_texture_model(self):
        if 'texture' not in self.models:
            logger.info('==> Loading texture model ...')
            texture_device = self.config['texture'].get('device', 'cuda:1')
            texture_model_config = OmegaConf.load(self.config['texture']['model_config'])
            texture_model = instantiate_from_config(texture_model_config.model_config)
            model_ckpt_path = hf_hub_download(repo_id="LutaoJiang/DiMeR", filename="DiMeR_texture.ckpt", repo_type="model")
            state_dict = torch.load(model_ckpt_path, map_location='cuda:1')
            state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
            texture_model.load_state_dict(state_dict, strict=True)
            texture_model.to(texture_device)
            texture_model.eval()
            
            self.models['texture'] = texture_model
        return self.models['texture']

    def unload_model(self, model_name):
        print (f"Unloading model: {model_name}")
        if model_name in self.models:
            #if isinstance(self.models[model_name], tuple):
            #    for model in self.models[model_name]:
            #        model.to('cpu')
            #else:
            #    self.models[model_name].to('cpu')
            del self.models[model_name]
            gc.collect()
            torch.cuda.empty_cache()

class kiss3d_wrapper:
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.config = model_loader.config
        self.to_512_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((512, 512), interpolation=2),
        ])
        self.renew_uuid()

    def renew_uuid(self):
        self.uuid = uuid.uuid4()

    def context(self):
        return torch.no_grad()

    def reconstruct_3d_bundle_image(self, image, camera_radius=3.5, lrm_render_radius=4.15, isomer_radius=4.5, 
                                  reconstruction_stage1_steps=0, reconstruction_stage2_steps=20, save_intermediate_results=True):
        recon_model = self.model_loader.load_reconstruction_model()
        recon_device = self.config['reconstruction'].get('device', 'cuda:1')

        # Load the reconstruction config file
        recon_config = OmegaConf.load(self.config['reconstruction']['model_config'])

        # split rgb and normal
        images = rearrange(image, 'c (n h) (m w) -> (n m) c h w', n=2, m=4)
        rgb_multi_view, normal_multi_view = images.chunk(2, dim=0)
        multi_view_mask = get_background(normal_multi_view).to(recon_device)
        rgb_multi_view = rgb_multi_view.to(recon_device) * multi_view_mask + (1 - multi_view_mask)
        
        with self.context():
            result = DiMeR_reconstruct(recon_model, recon_config,
                                    self.model_loader.load_texture_model(), self.config['texture']['model_config'],
                                    rgb_multi_view.to(recon_device), normal_multi_view.to(recon_device), 
                                    name=self.uuid, input_camera_type='kiss3d', 
                                    render_3d_bundle_image=save_intermediate_results,
                                    render_azimuths=[0, 90, 180, 270],
                                    render_elevations=[5, 5, 5, 5],
                                    render_radius=lrm_render_radius,
                                    camera_radius=camera_radius)
        
        self.model_loader.unload_model('reconstruction')
        self.model_loader.unload_model('texture')
        return result

    def generate_3d_bundle_image_normals(self, 
                                  image=None,
                                  save_intermediate_results=True):

        normals_pipe = self.model_loader.load_normals_model()

        if image is None:
            image = torch.zeros((1, 3, 1024, 2048), dtype=torch.float32, device="cuda:0")
        else:
            # Convert tensor to PIL Image if needed
            if isinstance(image, torch.Tensor):
                # Remove batch dimension if present
                if image.dim() == 4:
                    image = image.squeeze(0)  # Remove batch dimension
                
                # Split into individual views
                width = image.shape[2] // 4  # Each view is 1/4 of the width
                views = []
                for i in range(4):
                    view = image[:, :, i*width:(i+1)*width]
                    # Convert to PIL Image
                    view = view.permute(1, 2, 0).mul(255).byte().cpu().numpy()
                    views.append(Image.fromarray(view))

        with self.context():
            # Process each view separately
            normal_views = []
            for view in views:
                normal_view = normals_pipe(view, data_type="object")
                normal_views.append(normal_view)

        # Convert normal views to tensors and combine
        normal_tensors = []
        for normal_view in normal_views:
            normal_tensor = torchvision.transforms.functional.to_tensor(normal_view)
            normal_tensors.append(normal_tensor)
        
        # Stack normal tensors horizontally
        normal_images = torch.stack(normal_tensors)
        normal_grid = torchvision.utils.make_grid(normal_images, nrow=4, padding=0)

        # Create the final bundle by stacking RGB and normal grids vertically
        # First, create a grid of the original RGB images
        rgb_grid = torchvision.utils.make_grid(image.unsqueeze(0), nrow=4, padding=0)
        
        # Stack RGB and normal grids vertically
        bundle_image = torch.cat([rgb_grid, normal_grid], dim=1)  # Stack vertically

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f'{self.uuid}_gen_3d_bundle_image.png')
            torchvision.utils.save_image(bundle_image, save_path)
            logger.info(f"Save generated 3D bundle image to {save_path}")
            return bundle_image, save_path

        self.model_loader.unload_model('normals')
        return bundle_image

    def preprocess_controlnet_cond_image(self, image, mode, down_scale=1, kernel_size=51, sigma=2.0):
        """Preprocess image for controlnet conditioning"""
        if mode == 'tile':
            return self.to_512_tensor(image).unsqueeze(0)
        else:
            raise NotImplementedError(f"Control mode {mode} not implemented")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")

def image2mesh_main(k3d_wrapper, input_image, reference_3d_bundle_image, strength1=0.5, strength2=0.95, enable_redux=True):
    #seed_everything(seed)

    if enable_redux:
        redux_hparam = {
            'image': k3d_wrapper.to_512_tensor(input_image).unsqueeze(0).clip(0., 1.),
            'prompt_embeds_scale': 1.0,
            'pooled_prompt_embeds_scale': 1.0,
            'strength': strength1
        }
    else:
        redux_hparam = None

    # Extract just the RGB images (top row) from the bundle
    height = reference_3d_bundle_image.shape[1] // 2  # Get half the height
    rgb_images = reference_3d_bundle_image[:, :height, :]  # Take top half
    
    # Convert tensor back to PIL Image for preprocessing
    reference_pil = torchvision.transforms.ToPILImage()(rgb_images)
    
    gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_normals(
        image=rgb_images.unsqueeze(0),
    )

    # recon from 3D Bundle image
    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(gen_3d_bundle_image, save_intermediate_results=True)

    return gen_save_path, recon_mesh_path

def init_wrapper_from_config(config_path):
    model_loader = ModelLoader(config_path)
    return kiss3d_wrapper(model_loader) 