from PIL import Image,ImageSequence
import numpy as np
import torch
from moviepy.editor import VideoFileClip
import os
import imageio
import cv2
import random
from diffusers.utils import  export_to_video
import torch
import torch.fft
from blendmodes.blend import BlendType, blendLayers
from skimage import exposure
from moviepy.editor import ImageSequenceClip
import torch
import torch.fft as fft
import math

def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)

    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)

def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results

def numpy2pytorch(imgs, device, dtype):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h.to(device=device, dtype=dtype)

def get_fg_video(video_list, mask_list, device, dtype):
    video_np = np.stack(video_list, axis=0)
    mask_np = np.stack(mask_list, axis=0)
    mask_bool = mask_np == 255
    video_fg = np.where(mask_bool, video_np, 127)

    h = torch.from_numpy(video_fg).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h.to(device=device, dtype=dtype)


def pad(x, p, i):
    return x[:i] if len(x) >= i else x + [p] * (i - len(x))


def tensor2vid(video: torch.Tensor, processor, output_type="np"):

    batch_size, channels, num_frames, height, width = video.shape ## [1, 4, 16, 512, 512]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs

def read_video(video_path:str, image_width, image_height):
    extension = video_path.split('.')[-1].lower()
    video_name = os.path.basename(video_path)
    video_list = []

    if extension in "gif":
        ## input from gif
        video = Image.open(video_path)
        for i, frame in enumerate(ImageSequence.Iterator(video)):
            frame = np.array(frame.convert("RGB"))
            frame = resize_and_center_crop(frame, image_width, image_height)
            video_list.append(frame)
    elif extension in "mp4":
        ## input from mp4
        reader = imageio.get_reader(video_path)
        for frame in reader:
            frame = resize_and_center_crop(frame, image_width, image_height)
            video_list.append(frame)
    else:
        raise ValueError('Wrong input type')

    video_list = [Image.fromarray(frame) for frame in video_list]

    return video_list, video_name

def read_delight_video(video_path:str, image_width, image_height):
    extension = video_path.split('.')[-1].lower()
    video_name = os.path.basename(video_path)
    video_list = []


    # Create predictor instance
    # predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)
    predictor = torch.hub.load("/workspace/pyz/StableDelight", "StableDelight_turbo", source="local", trust_repo=True)


    if extension in "gif":
        ## input from gif
        video = Image.open(video_path)
        for i, frame in enumerate(ImageSequence.Iterator(video)):
            # Apply the model to the image
            frame = predictor(frame)
            # Delight frame
            frame = np.array(frame.convert("RGB"))
            frame = resize_and_center_crop(frame, image_width, image_height)
            video_list.append(frame)
    elif extension in "mp4":
        ## input from mp4
        reader = imageio.get_reader(video_path)
        for i,frame in enumerate(reader):
            # Apply the model to the image
            frame = Image.fromarray(frame)
            frame = predictor(frame)
            # Delight frame
            # Save or display the result
            filename = os.path.join("output/test_delight/delight", f"{video_name}_{i}.png")
            frame.save(filename)
            frame = np.array(frame)
            frame = resize_and_center_crop(frame, image_width, image_height)
            video_list.append(frame)
    else:
        raise ValueError('Wrong input type')

    video_list = [Image.fromarray(frame) for frame in video_list]
    return video_list, video_name

def read_normal_video(video_path:str, image_width, image_height):
    extension = video_path.split('.')[-1].lower()
    video_name = os.path.basename(video_path)
    video_list = []

    predictor = torch.hub.load("/workspace/pyz/StableNormal", "StableNormal_turbo", source="local",trust_repo=True)

    if extension in "gif":
        ## input from gif
        video = Image.open(video_path)
        for i, frame in enumerate(ImageSequence.Iterator(video)):
            # Apply the model to the image
            frame = predictor(frame)
            # Delight frame
            frame = np.array(frame.convert("RGB"))
            frame = resize_and_center_crop(frame, image_width, image_height)
            video_list.append(frame)
    elif extension in "mp4":
        ## input from mp4
        reader = imageio.get_reader(video_path)
        for i,frame in enumerate(reader):
            # Apply the model to the image
            frame = Image.fromarray(frame)
            frame = predictor(frame)

            frame = np.array(frame)
            frame = resize_and_center_crop(frame, image_width, image_height)
            video_list.append(frame)
    else:
        raise ValueError('Wrong input type')

    video_list = [Image.fromarray(frame) for frame in video_list]
    return video_list, video_name


def read_mask(mask_folder:str):
    mask_files = os.listdir(mask_folder)
    mask_files = sorted(mask_files)
    mask_list = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        mask = Image.open(mask_path).convert('RGB')
        mask_list.append(mask)
    
    return mask_list

def decode_latents(vae, latents, decode_chunk_size: int = 16):
    
    latents = 1 / vae.config.scaling_factor * latents
    video = []
    for i in range(0, latents.shape[0], decode_chunk_size):
        batch_latents = latents[i : i + decode_chunk_size]
        batch_latents = vae.decode(batch_latents).sample
        video.append(batch_latents)

    video = torch.cat(video)

    return video

def encode_video(vae, video, decode_chunk_size: int = 16) -> torch.Tensor:
    latents = []
    for i in range(0, len(video), decode_chunk_size):
        batch_video = video[i : i + decode_chunk_size]
        batch_video = vae.encode(batch_video).latent_dist.mode()
        latents.append(batch_video)
    return torch.cat(latents)

def vis_video(input_video, video_processor, save_path):
    ## shape: 1, c, f, h, w
    relight_video = video_processor.postprocess_video(video=input_video, output_type="pil")
    export_to_video(relight_video[0], save_path)
    
def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def setup_color_correction(image):
    correction_target = cv2.cvtColor(np.asarray(image.copy()),
                                     cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    image = Image.fromarray(
        cv2.cvtColor(
            exposure.match_histograms(cv2.cvtColor(np.asarray(original_image),
                                                   cv2.COLOR_RGB2LAB),
                                      correction,
                                      channel_axis=2),
            cv2.COLOR_LAB2RGB).astype('uint8'))

    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image


def save_tensor_to_video_moviepy(tensor, output_dir, i,inversion_video_name, fps=25):
    # 生成带序号的输出路径
    output_path = os.path.join(output_dir, f"{inversion_video_name}Denoising_step_{i}.mp4")
    # 检查输入张量的形状是否为 (F, C, H, W)
    if len(tensor.shape) != 4:
        raise ValueError("输入的张量形状必须为 (F, C, H, W)")
    F, C, H, W = tensor.shape
    # 确保帧率为整数
    fps = int(fps)
    frames = []
    for frame in tensor:
        # 将张量转换为 numpy 数组，并调整通道顺序
        frame = frame.permute(1, 2, 0).cpu().numpy()
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frames.append(frame)
    # 创建视频剪辑
    clip = ImageSequenceClip(frames, fps=fps)
    # 保存视频
    clip.write_videofile(output_path)



def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain
    # x_freq_mixed = lbd * x_freq_low + (1-lbd) * noise_freq_high
    # x_freq_mixed = (1-lbd) *x_freq_low + lbd *  noise_freq_high
    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed



def get_freq_filter(shape, device, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask