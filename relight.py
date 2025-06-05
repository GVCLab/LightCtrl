import os
import torch
import imageio
import argparse
import numpy as np
from types import MethodType
import safetensors.torch as sf
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import MotionAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from torch.hub import download_url_to_file
from PIL import Image
from models.scheduling_ddim import DDIMScheduler  # add reverse step

from src.ic_light import BGSource

####################################################################
# from src.animatediff_pipe import AnimateDiffVideoToVideoPipeline
from src.vdm_pipe import AnimateDiffVideoToVideoPipeline
####################################################################

from src.ic_light_pipe import StableDiffusionImg2ImgPipeline
from utils.tools import read_video, set_all_seed



def main(args):
    
    config  = OmegaConf.load(args.config)
    device = torch.device('cuda')
    adopted_dtype = torch.float16
    set_all_seed(42)
    
    ## vdm model
    adapter = MotionAdapter.from_pretrained(args.motion_adapter_model)

    ## pipeline
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(args.sd_model, motion_adapter=adapter)
    eul_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        args.sd_model,
        subfolder="scheduler",
        beta_schedule="linear",
    )

#     inversion_scheduler = DDIMScheduler.from_pretrained("/workspace/pyz/IC-Light/models/stablediffusionapi-realistic-vision-v51", subfolder="scheduler")
#     # sd_path="/workspace/pyz/IC-Light/models/stablediffusionapi-realistic-vision-v51"
#     inversion_scheduler = DDIMScheduler(
#     num_train_timesteps=1000,
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     clip_sample=False,
#     set_alpha_to_one=False,
#     steps_offset=1,
# )

    pipe.scheduler = eul_scheduler
    # pipe.scheduler = inversion_scheduler
    pipe.enable_vae_slicing()
    pipe = pipe.to(device=device, dtype=adopted_dtype)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    ## ic-light model
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet")
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_() 
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    unet_original_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    unet.forward = hooked_unet_forward

    ## ic-light model loader
    if not os.path.exists(args.ic_light_model):
        download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', 
                             dst=args.ic_light_model)
    
    sd_offset = sf.load_file(args.ic_light_model)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged
    text_encoder = text_encoder.to(device=device, dtype=adopted_dtype)
    vae = vae.to(device=device, dtype=adopted_dtype)
    unet = unet.to(device=device, dtype=adopted_dtype)
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())


    ## ic-light-scheduler
    ic_light_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1
    )

    

    ic_light_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=ic_light_scheduler,
        # scheduler=inversion_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )
    ic_light_pipe = ic_light_pipe.to(device)

    
    #############################  params  ######################################
    strength = config.get("strength", 0.5)
    num_step = config.get("num_step", 20)
    text_guide_scale = config.get("text_guide_scale", 7.0)
    seed = config.get("seed")
    image_width = config.get("width", 512)
    image_height = config.get("height", 512)
    n_prompt = config.get("n_prompt", "")
    relight_prompt = config.get("relight_prompt", "")
    video_path = config.get("video_path", "")
    save_path = config.get("save_path")

    ##############################  infer  #####################################
    generator = torch.manual_seed(seed)
    video_name = os.path.basename(video_path)
    video_list, video_name = read_video(video_path, image_width, image_height)

    print("################## begin ##################")
    with torch.no_grad():
        num_inference_steps = int(round(num_step / strength))
        
        output = pipe(
            ic_light_pipe=ic_light_pipe,
            relight_prompt=relight_prompt,
            video=video_list,
            video_path=video_path,
            prompt=relight_prompt,
            strength=strength,
            negative_prompt=n_prompt,
            guidance_scale=text_guide_scale,
            num_inference_steps=num_inference_steps,
            height=image_height,
            width=image_width,
            generator=generator,
        )

        frames = output.frames[0]#List
        results_path = f"{save_path}/relight_{video_name}"
        imageio.mimwrite(results_path, frames, fps=8)

        print(f"relight finished! save in {results_path}.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sd_model", type=str, default="/workspace/pyz/IC-Light/models/stablediffusionapi-realistic-vision-v51")
    parser.add_argument("--motion_adapter_model", type=str, default="/workspace/pyz/.cache/huggingface/hub/models--guoyww--animatediff-motion-adapter-v1-5-3/snapshots/animatediff-motion-adapter-v1-5-3")
    parser.add_argument("--ic_light_model", type=str, default="/workspace/pyz/LAV-iclight/models/iclight_sd15_fc.safetensors")
    
    parser.add_argument("--config", type=str, default="configs/relight/car.yaml", help="the config file for each sample.")
    
    args = parser.parse_args()
    main(args)