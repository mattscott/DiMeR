import os
import uuid
import torch
import torchvision
import numpy as np
import yaml
import logging
import random

from PIL import Image
from einops import rearrange
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict, Union, Any
from omegaconf import OmegaConf
from models.DiMeR.utils.train_util import instantiate_from_config
from huggingface_hub import hf_hub_download
access_token = os.getenv("HUGGINGFACE_TOKEN")

from pipeline.utils import logger, TMP_DIR, OUT_DIR
from pipeline.utils import lrm_reconstruct, isomer_reconstruct, preprocess_input_image, DiMeR_reconstruct
from utils.tool import get_background

from pipeline.custom_pipelines import FluxPriorReduxPipeline, FluxControlNetImg2ImgPipeline, FluxImg2ImgPipeline
from diffusers import FluxPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler, FluxTransformer2DModel, AutoencoderTiny
from diffusers.models.controlnets.controlnet_flux import FluxMultiControlNetModel, FluxControlNetModel
from diffusers.schedulers import FlowMatchHeunDiscreteScheduler

logger = logging.getLogger(__name__)

def convert_flux_pipeline(exist_flux_pipe, target_pipe, **kwargs):
    new_pipe = target_pipe(
        scheduler = exist_flux_pipe.scheduler,
        vae = exist_flux_pipe.vae,
        text_encoder = exist_flux_pipe.text_encoder,
        tokenizer = exist_flux_pipe.tokenizer,
        text_encoder_2 = exist_flux_pipe.text_encoder_2,
        tokenizer_2 = exist_flux_pipe.tokenizer_2,
        transformer = exist_flux_pipe.transformer,
        **kwargs
    )
    return new_pipe

def init_wrapper_from_config(config_path):
    with open(config_path, 'r') as config_file:
        config_ = yaml.safe_load(config_file)

    dtype_ = {
        'fp8': torch.float8_e4m3fn,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }
    
    # init flux_pipeline
    logger.info('==> Loading Flux model ...')
    flux_device = config_['flux'].get('device', 'cpu')
    flux_base_model_pth = config_['flux'].get('base_model', None)
    flux_dtype = config_['flux'].get('dtype', 'bf16')
    flux_controlnet_pth = config_['flux'].get('controlnet', None)
    # flux_lora_pth = config_['flux'].get('lora', None)
    flux_lora_pth = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="rgb_normal.safetensors", repo_type="model", token=access_token)
    flux_redux_pth = config_['flux'].get('redux', None)
    # taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype_[flux_dtype]).to(flux_device)
    if flux_base_model_pth.endswith('safetensors'):
        flux_pipe = FluxImg2ImgPipeline.from_single_file(flux_base_model_pth, torch_dtype=dtype_[flux_dtype], token=access_token)
    else:
        flux_pipe = FluxImg2ImgPipeline.from_pretrained(flux_base_model_pth, torch_dtype=dtype_[flux_dtype], token=access_token)
    flux_pipe.vae.enable_slicing()
    flux_pipe.vae.enable_tiling()
    
    # load flux model and controlnet
    if flux_controlnet_pth is not None and False:
        flux_controlnet = FluxControlNetModel.from_pretrained(flux_controlnet_pth, torch_dtype=torch.bfloat16)
        flux_pipe = convert_flux_pipeline(flux_pipe, FluxControlNetImg2ImgPipeline, controlnet=[flux_controlnet])

    flux_pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(flux_pipe.scheduler.config)
        
    # load lora weights
    flux_pipe.load_lora_weights(flux_lora_pth)
    # flux_pipe.to(device=flux_device)

    # load redux model
    flux_redux_pipe = None
    if flux_redux_pth is not None and False:
        flux_redux_pipe = FluxPriorReduxPipeline.from_pretrained(flux_redux_pth, torch_dtype=torch.bfloat16, token=access_token)
        flux_redux_pipe.text_encoder = flux_pipe.text_encoder
        flux_redux_pipe.text_encoder_2 = flux_pipe.text_encoder_2
        flux_redux_pipe.tokenizer = flux_pipe.tokenizer
        flux_redux_pipe.tokenizer_2 = flux_pipe.tokenizer_2
        # flux_redux_pipe.to(device=flux_device)

    # Initialize caption model
    caption_config = config_['caption']
    caption_device = caption_config.get('device', 'cuda:0')
    caption_processor = AutoProcessor.from_pretrained(caption_config['base_model'], trust_remote_code=True)
    caption_model = AutoModelForCausalLM.from_pretrained(caption_config['base_model'], trust_remote_code=True)
    caption_model.to(caption_device)

    # Initialize reconstruction model
    logger.info('==> Loading reconstruction model ...')
    #recon_model_config = config_['reconstruction']
    #recon_device = recon_model_config.get('device', 'cuda:0')
    #recon_model = recon_model_config['model'](recon_model_config['model_config'])
    #recon_model.to(recon_device)

    recon_device = config_['reconstruction'].get('device', 'cuda:0')
    recon_model_config = OmegaConf.load(config_['reconstruction']['model_config'])
    recon_model = instantiate_from_config(recon_model_config.model_config)
    model_ckpt_path = hf_hub_download(repo_id="LutaoJiang/DiMeR", filename="DiMeR_geometry.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location='cuda:0')
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    recon_model.load_state_dict(state_dict, strict=True)
    recon_model.to(recon_device)
    recon_model.eval()

    # Initialize texture model
    logger.info('==> Loading texture model ...')
    #texture_model_config = config_['texture']
    #texture_device = texture_model_config.get('device', 'cuda:0')
    #texture_model = texture_model_config['model'](texture_model_config['model_config'])
    #texture_model.to(texture_device)

    texture_device = config_['texture'].get('device', 'cuda:0')
    texture_model_config = OmegaConf.load(config_['texture']['model_config'])
    texture_model = instantiate_from_config(texture_model_config.model_config)
    # load recon model checkpoint
    model_ckpt_path = hf_hub_download(repo_id="LutaoJiang/DiMeR", filename="DiMeR_texture.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location='cuda:0')
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    texture_model.load_state_dict(state_dict, strict=True)
    texture_model.to(texture_device)
    texture_model.eval()

    return kiss3d_wrapper(
        config=config_,
        flux_pipeline = flux_pipe,
        flux_redux_pipeline=flux_redux_pipe,
        caption_processor=caption_processor,
        caption_model=caption_model,
        reconstruction_model_config=recon_model_config,
        reconstruction_model=recon_model,
        texture_model_config=texture_model_config,
        texture_model=texture_model
    )

class kiss3d_wrapper(object):
    def __init__(self,
        config: Dict,
        flux_pipeline: Union[FluxPipeline, FluxControlNetImg2ImgPipeline],
        flux_redux_pipeline: FluxPriorReduxPipeline,
        caption_processor: AutoProcessor,
        caption_model: AutoModelForCausalLM,
        reconstruction_model_config: Any,
        reconstruction_model: Any,
        texture_model_config: Any,
        texture_model: Any
    ):
        self.config = config
        self.flux_pipeline = flux_pipeline
        self.flux_redux_pipeline = flux_redux_pipeline
        self.caption_model = caption_model
        self.caption_processor = caption_processor
        self.recon_model_config = reconstruction_model_config
        self.recon_model = reconstruction_model
        self.texture_model_config = texture_model_config
        self.texture_model = texture_model

        self.to_512_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((512, 512), interpolation=2),
        ])

        self.renew_uuid()

    def renew_uuid(self):
        self.uuid = uuid.uuid4()

    def context(self):
        return torch.no_grad()

    def get_image_caption(self, image):
        """Generate a caption for the input image"""
        caption_device = self.config['caption'].get('device', 'cuda:0')
        self.caption_model.to(caption_device)
        
        with self.context():
            inputs = self.caption_processor(images=image, return_tensors="pt").to(caption_device)
            outputs = self.caption_model.generate(**inputs, max_new_tokens=50)
            caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
        
        self.caption_model.to('cuda:0')
        return caption

    def reconstruct_3d_bundle_image(self, 
        image, 
        camera_radius=3.5,
        lrm_render_radius=4.15, 
        isomer_radius=4.5, 
        reconstruction_stage1_steps=0,
        reconstruction_stage2_steps=20,
        save_intermediate_results=True):
        """
        image: torch.Tensor, range [0., 1.], (3, 1024, 2048)
        """
        recon_device = self.config['reconstruction'].get('device', 'cuda:0')

        # split rgb and normal
        images = rearrange(image, 'c (n h) (m w) -> (n m) c h w', n=2, m=4) # (3, 1024, 2048) -> (8, 3, 512, 512)
        rgb_multi_view, normal_multi_view = images.chunk(2, dim=0)
        multi_view_mask = get_background(normal_multi_view).to(recon_device)
        rgb_multi_view = rgb_multi_view.to(recon_device) * multi_view_mask + (1 - multi_view_mask)
        
        with self.context():
            return DiMeR_reconstruct(self.recon_model, self.recon_model_config.infer_config,
                              self.texture_model, self.texture_model_config.infer_config,
                            rgb_multi_view.to(recon_device), normal_multi_view.to(recon_device), name=self.uuid, 
                            input_camera_type='kiss3d', render_3d_bundle_image=save_intermediate_results,
                            render_azimuths=[0, 90, 180, 270],
                            render_radius=lrm_render_radius,
                            camera_radius=camera_radius)

    def generate_3d_bundle_image_controlnet(self, 
                                 prompt, 
                                 image=None,
                                 strength=1.0, 
                                 control_image=[],
                                 control_mode=[],
                                 control_guidance_start=None,
                                 control_guidance_end=None,
                                 controlnet_conditioning_scale=None,
                                 lora_scale=1.0,
                                 num_inference_steps=None,
                                 seed=None,
                                 redux_hparam=None,
                                 save_intermediate_results=True,
                                 **kwargs):
        control_mode_dict = {
            'canny': 0,
            'tile': 1,
            'depth': 2,
            'blur': 3,
            'pose': 4,
            'gray': 5,
            'lq': 6,
        }

        flux_device = self.config['flux'].get('device', 'cuda:0')
        self.flux_pipeline.to(flux_device)
        seed = seed or self.config['flux'].get('seed', 0)
        num_inference_steps = num_inference_steps or self.config['flux'].get('num_inference_steps', 20)

        generator = torch.Generator(device=flux_device).manual_seed(seed)

        if image is None:
            image = torch.zeros((1, 3, 1024, 2048), dtype=torch.float32, device=flux_device)

        hparam_dict = {
            'prompt': prompt,
            'image': image,
            'strength': strength,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': 3.5,
            'num_images_per_prompt': 1,
            'width': 2048,
            'height': 1024,
            'output_type': 'np',
            'generator': generator,
            'joint_attention_kwargs': {"scale": lora_scale}
        }
        hparam_dict.update(kwargs)

        # append controlnet hparams
        if len(control_image) > 0:
            assert len(control_mode) == len(control_image)
            
            ctrl_hparams = {
                'control_mode': [control_mode_dict[mode_] for mode_ in control_mode],
                'control_image': control_image,
                'control_guidance_start': control_guidance_start or [0.0 for i in range(len(control_image))],
                'control_guidance_end': control_guidance_end or [1.0 for i in range(len(control_image))],
                'controlnet_conditioning_scale': controlnet_conditioning_scale or [1.0 for i in range(len(control_image))],
            }

            hparam_dict.update(ctrl_hparams)

        with self.context():
            gen_3d_bundle_image = self.flux_pipeline(**hparam_dict).images
        
        gen_3d_bundle_image_ = torch.from_numpy(gen_3d_bundle_image).squeeze(0).permute(2, 0, 1).contiguous().float()

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f'{self.uuid}_gen_3d_bundle_image.png')
            torchvision.utils.save_image(gen_3d_bundle_image_, save_path)
            logger.info(f"Save generated 3D bundle image to {save_path}")
            return gen_3d_bundle_image_, save_path

        return gen_3d_bundle_image_

    def generate_3d_bundle_image_text(self, 
                                    prompt,
                                    image=None, 
                                    strength=1.0,
                                    lora_scale=1.0,
                                    num_inference_steps=None,
                                    seed=None,
                                    redux_hparam=None,
                                    save_intermediate_results=True,
                                    **kwargs):
        flux_device = self.config['flux'].get('device', 'cuda:0')
        seed = seed or self.config['flux'].get('seed', 0)
        num_inference_steps = num_inference_steps or self.config['flux'].get('num_inference_steps', 20)

        generator = torch.Generator(device=flux_device).manual_seed(seed)

        hparam_dict = {
            'prompt': 'A grid of 2x4 multi-view image, elevation 5. White background.',
            'prompt_2': ' '.join(['A grid of 2x4 multi-view image, elevation 5. White background.', prompt]),
            'image': image,
            'strength': strength,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': 3.5,
            'num_images_per_prompt': 1,
            'width': 2048,
            'height': 1024,
            'output_type': 'np',
            'generator': generator,
            'joint_attention_kwargs': {"scale": lora_scale}
        }
        hparam_dict.update(kwargs)

        with self.context():
            gen_3d_bundle_image = self.flux_pipeline(**hparam_dict).images

        gen_3d_bundle_image_ = torch.from_numpy(gen_3d_bundle_image).squeeze(0).permute(2, 0, 1).contiguous().float()

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f'{self.uuid}_gen_3d_bundle_image.png')
            torchvision.utils.save_image(gen_3d_bundle_image_, save_path)
            logger.info(f"Save generated 3D bundle image to {save_path}")
            return gen_3d_bundle_image_, save_path

        return gen_3d_bundle_image_

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

def image2mesh_main(k3d_wrapper, input_image, reference_3d_bundle_image, caption, seed, strength1=0.5, strength2=0.95, enable_redux=True, use_controlnet=True):
    seed_everything(seed)

    if enable_redux:
        redux_hparam = {
            'image': k3d_wrapper.to_512_tensor(input_image).unsqueeze(0).clip(0., 1.),
            'prompt_embeds_scale': 1.0,
            'pooled_prompt_embeds_scale': 1.0,
            'strength': strength1
        }
    else:
        redux_hparam = None

    if use_controlnet:
        # Convert tensor back to PIL Image for preprocessing
        reference_pil = torchvision.transforms.ToPILImage()(reference_3d_bundle_image)
        
        # prepare controlnet condition
        control_mode = ['tile']
        control_image = [k3d_wrapper.preprocess_controlnet_cond_image(reference_pil, mode_, down_scale=1, kernel_size=51, sigma=2.0) for mode_ in control_mode]
        control_guidance_start = [0.0]
        control_guidance_end = [0.3]
        controlnet_conditioning_scale = [0.3]

        gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_controlnet(
            prompt=caption,
            image=reference_3d_bundle_image.unsqueeze(0),
            strength=strength2,
            control_image=control_image, 
            control_mode=control_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            lora_scale=1.0,
            redux_hparam=redux_hparam
        )
    else:
        gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_text(
            prompt=caption,
            image=reference_3d_bundle_image.unsqueeze(0),
            strength=strength2,
            lora_scale=1.0,
            redux_hparam=redux_hparam
        )

    # recon from 3D Bundle image
    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(gen_3d_bundle_image, save_intermediate_results=False)

    return gen_save_path, recon_mesh_path 