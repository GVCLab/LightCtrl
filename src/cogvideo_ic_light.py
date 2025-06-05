import torch
import numpy as np
from enum import Enum
import math
import cv2
import os
import imageio
from einops import rearrange
from torchvision import transforms
from diffusers.video_processor import VideoProcessor
import torch.nn.functional as F
from utils.tools import resize_and_center_crop, numpy2pytorch, decode_latents, encode_video,get_freq_filter, freq_mix_3d,pad


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    TOP_LEFT_TO_BOTTOM_RIGHT = "TOP_LEFT_TO_BOTTOM_RIGHT Light"
    BOTTOM_LEFT_TO_TOP_RIGHT = "BOTTOM_LEFT_TO_TOP_RIGHT Light"
    TOP_TO_BOTTOM = "TOP_TO_BOTTOM Light"
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT Light"
    CIRCULAR = "CIRCULAR Light"
    SMI_CIRCULAR = "smi CIRCULAR Light"
    RADIUS_CHANGE = "R_CHANGE Light"
    TOP_RIGHT_BOTTOM_LEFT_CYCLE = "top_right_bottom_left_cycle"
    LEFT_TOP_RIGHT_BOTTOM_CYCLE = "left_top_right_bottom_cycle"
    BOTTOM_LEFT_TOP_RIGHT_CYCLE = "bottom_left_top_right_cycle"
    RIGHT_BOTTOM_LEFT_TOP_CYCLE = "right_bottom_left_top_cycle"





class Relighter:
    def __init__(self, 
                 delight_target,
                 lbd,
                 pipeline, 
                 relight_prompt="",
                 num_frames=16,
                 image_width=512,
                 image_height=512, 
                 num_samples=1, 
                 steps=15, 
                 cfg=2, 
                 lowres_denoise=0.9, 
                 
                 generator=None,
                #  fg_video=None,
                 ):
        is_code_run = False
        self.pipeline = pipeline
        self.delight_target =delight_target
        self.image_width = image_width
        self.image_height = image_height
        self.num_samples = num_samples
        self.steps = steps
        self.cfg = cfg
        self.lowres_denoise = lowres_denoise

        self.generator = generator
        self.device = pipeline.device
        self.num_frames = num_frames
        self.vae = self.pipeline.vae
        
        self.a_prompt = "best quality"
        self.n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
        positive_prompt = relight_prompt + ', ' + self.a_prompt
        negative_prompt = self.n_prompt
        tokenizer = self.pipeline.tokenizer
        device = self.pipeline.device
        vae = self.vae
        
        conds, unconds = self.encode_prompt_pair(tokenizer, device, positive_prompt, negative_prompt)

        #diverse Light map
        if not is_code_run:
             # 生成 TOP_LEFT_TO_BOTTOM_RIGHT 的 16 帧序列
            frames_top_left_to_bottom_right = []
            for i in range(self.num_frames):
                input_bg = self.create_dynamic_background(BGSource.TOP_LEFT_TO_BOTTOM_RIGHT, frame_index=i, total_frames=self.num_frames)
                filename = os.path.join("output/top_left_to_bottom_right_bg", f"{i}.png")
                cv2.imwrite(filename, input_bg)
                bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)  # 调整背景大小
                bg_latent = numpy2pytorch([bg], device, vae.dtype)  # 转换为 PyTorch 张量
                bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor  # 编码为 latent
                frames_top_left_to_bottom_right.append(bg_latent)

            # 生成 BOTTOM_LEFT_TO_TOP_RIGHT 的 self.num_frames 帧序列
            frames_bottom_left_to_top_right = []
            for i in range(self.num_frames):
                input_bg = self.create_dynamic_background(BGSource.BOTTOM_LEFT_TO_TOP_RIGHT, frame_index=i, total_frames=self.num_frames)
                filename = os.path.join("output/bottom_left_to_top_right_bg", f"{i}.png")
                cv2.imwrite(filename, input_bg)
                bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)  # 调整背景大小
                bg_latent = numpy2pytorch([bg], device, vae.dtype)  # 转换为 PyTorch 张量
                bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor  # 编码为 latent
                frames_bottom_left_to_top_right.append(bg_latent)

            # 生成 circular 的 self.num_frames 帧序列
            frames_circular = []
            for i in range(self.num_frames):
                input_bg = self.create_dynamic_background(BGSource.CIRCULAR, frame_index=i, total_frames=self.num_frames)
                filename = os.path.join("output/circular_bg", f"{i}.png")
                cv2.imwrite(filename, input_bg)
                bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)  # 调整背景大小
                bg_latent = numpy2pytorch([bg], device, vae.dtype)  # 转换为 PyTorch 张量
                bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor  # 编码为 latent
                frames_circular.append(bg_latent)

            # 生成 circular 的 self.num_frames 帧序列
            frames_smi_circular = []
            for i in range(self.num_frames):
                input_bg = self.create_dynamic_background(BGSource.SMI_CIRCULAR, frame_index=i, total_frames=self.num_frames)
                filename = os.path.join("output/smi_circular_bg", f"{i}.png")
                cv2.imwrite(filename, input_bg)
                bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)  # 调整背景大小
                bg_latent = numpy2pytorch([bg], device, vae.dtype)  # 转换为 PyTorch 张量
                bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor  # 编码为 latent
                frames_smi_circular.append(bg_latent)

            # 生成 TOP-TO-BOTTOM 的 self.num_frames 帧序列
            frames_top_to_bottom = []
            for i in range(self.num_frames):
                input_bg = self.create_dynamic_background(BGSource.TOP_TO_BOTTOM, frame_index=i, total_frames=self.num_frames)
                filename = os.path.join("output/top_to_bottom_bg", f"{i}.png")
                cv2.imwrite(filename, input_bg)
                bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)  # 调整背景大小
                bg_latent = numpy2pytorch([bg], device, vae.dtype)  # 转换为 PyTorch 张量
                bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor  # 编码为 latent
                frames_top_to_bottom.append(bg_latent)

            # 生成 LEFT-TO-RIGHT 的 self.num_frames 帧序列
            frames_left_to_right = []
            for i in range(self.num_frames):
                input_bg = self.create_dynamic_background(BGSource.LEFT_TO_RIGHT, frame_index=i, total_frames=self.num_frames)
                filename = os.path.join("output/left_to_right_bg", f"{i}.png")
                cv2.imwrite(filename, input_bg)
                bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)  # 调整背景大小
                bg_latent = numpy2pytorch([bg], device, vae.dtype)  # 转换为 PyTorch 张量
                bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor  # 编码为 latent
                frames_left_to_right.append(bg_latent)

            # 生成 LEFT-TO-RIGHT 的 self.num_frames 帧序列
            frames_radius_change = []
            for i in range(self.num_frames):
                input_bg = self.create_dynamic_background(BGSource.RADIUS_CHANGE, frame_index=i, total_frames=self.num_frames)
                filename = os.path.join("output/radius_change_bg", f"{i}.png")
                cv2.imwrite(filename, input_bg)
                bg = resize_and_center_crop(input_bg, self.image_width, self.image_height)  # 调整背景大小
                bg_latent = numpy2pytorch([bg], device, vae.dtype)  # 转换为 PyTorch 张量
                bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor  # 编码为 latent
                frames_radius_change.append(bg_latent)

            
            
            is_code_run = True
       

    


        #choice your light map trajectory
        self.bg_latent=frames_circular
        # self.bg_latent=frames_top_left_to_bottom_right
        # self.bg_latent=frames_bottom_left_to_top_right
        # self.bg_latent=frames_smi_circular
        # self.bg_latent=frames_left_to_right
        # self.bg_latent=frames_radius_change
        # self.bg_latent=frames_top_to_bottom

        self.bg_latent = torch.cat(self.bg_latent, dim=0)  # 形状为 [16, C, H, W]

        self.conds = conds.repeat(self.num_frames, 1, 1)
        self.unconds = unconds.repeat(self.num_frames, 1, 1)
        
    def encode_prompt_inner(self, tokenizer, txt):
        max_length = tokenizer.model_max_length
        chunk_length = tokenizer.model_max_length - 2
        id_start = tokenizer.bos_token_id
        id_end = tokenizer.eos_token_id
        id_pad = id_end

        tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = self.pipeline.text_encoder(token_ids).last_hidden_state
        return conds

    def encode_prompt_pair(self, tokenizer, device, positive_prompt, negative_prompt):
        c = self.encode_prompt_inner(tokenizer, positive_prompt)
        uc = self.encode_prompt_inner(tokenizer, negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c.to(device), uc.to(device)

    

    def create_dynamic_background(self, bg_source, frame_index, total_frames=16):

        max_pix = 255
        min_pix = 0

        # 计算当前帧的进度 (0 到 1)
        t = frame_index / (total_frames - 1)

        # 创建网格
        x = np.arange(self.image_width)
        y = np.arange(self.image_height)
        xx, yy = np.meshgrid(x, y)

        if bg_source == BGSource.TOP_LEFT_TO_BOTTOM_RIGHT:
            # 光源从左上移动到右下
            light_x = int(t * (self.image_width - 1))
            light_y = int(t * (self.image_height - 1))
        elif bg_source == BGSource.BOTTOM_LEFT_TO_TOP_RIGHT:
            # 光源从左下移动到右上
            light_x = int(t * (self.image_width - 1))
            light_y = int((1 - t) * (self.image_height - 1))
        elif bg_source == BGSource.TOP_TO_BOTTOM:
            # 光源从上到下
            light_x = self.image_width // 4  # 光源始终在中间的列
            light_y = int(t * (self.image_height - 1))
        elif bg_source == BGSource.LEFT_TO_RIGHT:
            # 光源从左到右
            light_x = int(t * (self.image_width - 1))
            light_y = self.image_height // 4  # 光源始终在中间的行
        elif bg_source == BGSource.CIRCULAR:
            # 环形移动的光源
            center_x = self.image_width // 2
            center_y = self.image_height // 2
            radius = min(self.image_width, self.image_height) // 2

            # 计算光源的当前位置（沿着圆形路径移动）
            angle = 2 * np.pi * t
            light_x = int(center_x + radius * np.cos(angle))
            light_y = int(center_y + radius * np.sin(angle))
        elif bg_source == BGSource.SMI_CIRCULAR:
            # 环形移动的光源
            center_x = self.image_width // 2
            center_y = self.image_height // 2
            radius = min(self.image_width, self.image_height) // 2

            # 计算光源的当前位置（沿着圆形路径移动）
            angle = 1 * np.pi * t
            light_x = int(center_x - radius * np.cos(angle))
            light_y = int(center_y + radius * np.sin(-angle))
        elif bg_source == BGSource.RADIUS_CHANGE:
            # 光源半径随时间变化
            center_x = self.image_width // 4 # 光源固定在中心
            center_y = self.image_height // 4

            # 初始半径和变化范围
            base_radius = 20  # 初始半径
            delta_radius = 200  # 半径变化范围
            light_radius = base_radius + int(delta_radius * t)  # 半径随时间变化

            # 计算每个像素到光源中心的距离
            distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

            # 使用高斯分布模拟光源亮度衰减，sigma 随半径变化
            sigma = light_radius  # 控制亮度衰减的范围
            gradient = max_pix * np.exp(-(distance ** 2) / (2 * sigma ** 2))

            # 将亮度限制在 [min_pix, max_pix] 范围内
            gradient = np.clip(gradient, min_pix, max_pix)
            # 将单通道图像扩展为 RGB 图像
            image = np.stack((gradient,) * 3, axis=-1).astype(np.uint8)
            return image
        else:
            raise ValueError("Invalid bg_source! Please use a valid BGSource enum value.")

        # 计算每个像素到光源的距离
        distance = np.sqrt((xx - light_x) ** 2 + (yy - light_y) ** 2)

        # 光源辐射半径
        light_radius = 75 


        sigma = light_radius  # 控制亮度衰减的范围
        gradient = max_pix * np.exp(-(distance ** 2) / (2 * sigma ** 2))

        # 将亮度限制在 [min_pix, max_pix] 范围内
        gradient = np.clip(gradient, min_pix, max_pix)
        # 将单通道图像扩展为 RGB 图像
        image = np.stack((gradient,) * 3, axis=-1).astype(np.uint8)
        return image

    
    @torch.no_grad()
    def __call__(self, input_video, delight_target, lbd,init_latent=None, input_strength=None):

        input_latent = encode_video(self.vae, input_video)* self.vae.config.scaling_factor#torch.Size([16, 4, 64, 64])
        delight_latent = encode_video(self.vae,delight_target)* self.vae.config.scaling_factor
        
        
        X_low = delight_latent
        X_high = input_latent

        ########################## dynamic FFT   ##########################
        f,c,h,w = X_high.shape
        filter_shape = [
            f,
            c, 
            h, 
            w
        ]

        freq_filter = get_freq_filter(
            filter_shape, 
            device = self.device, 
            filter_type='butterworth',
            n=4,
            d_s=lbd,
            d_t=lbd
        )
        input_latent = freq_mix_3d(X_low.to(dtype=torch.float32), X_high.to(dtype=torch.float32), LPF=freq_filter)
        ########################## dynamic FFT   ##########################

        if input_strength:
            light_strength = input_strength
        else:
            light_strength = self.lowres_denoise

        if not init_latent:
            init_latent = self.bg_latent

        latents = self.pipeline(
            image=init_latent,#torch.Size([16, 4, 64, 64])
            strength=light_strength,
            prompt_embeds=self.conds,
            negative_prompt_embeds=self.unconds,
            width=self.image_width,
            height=self.image_height,
            num_inference_steps=int(round(self.steps / self.lowres_denoise)),#15/0.9
            num_images_per_prompt=self.num_samples,
            generator=self.generator,
            output_type='latent',
            guidance_scale=self.cfg,
            cross_attention_kwargs={'concat_conds': input_latent},
        ).images.to(self.pipeline.vae.dtype)

        relight_video = decode_latents(self.vae, latents)
        return relight_video