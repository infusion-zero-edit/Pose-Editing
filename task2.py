import warnings
warnings.filterwarnings('ignore')

import subprocess, io, os, sys, time
import gradio as gr
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, './GroundingDINO')

import diffusers
import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import time
import torch
import argparse
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops, ImageFilter
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms
import argparse
import copy
from scipy.ndimage import binary_dilation

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino import _C

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config as lama_Config    

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

import signal
import json
from datetime import date, datetime, timedelta
from gevent import pywsgi
import base64 
from loguru import logger

import cv2
import numpy as np
import matplotlib
matplotlib.use('AGG')
plt = matplotlib.pyplot
  
# diffusers
import ast
import PIL
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download
from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
)


config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
sam_checkpoint = './checkpoints/sam_vit_h_4b8939.pth' 
output_dir = "outputs"
_GPU_INDEX = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(output_dir, exist_ok=True)
groundingdino_model = None
sam_device = None
sam_model = None
sam_predictor = None
sam_mask_generator = None
lama_cleaner_model= None

def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    

def load_image(image_path):
    # # load image
    if isinstance(image_path, PIL.Image.Image):
        image_pil = image_path
    else:
        image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
    
def show_mask(mask, ax):
    color = np.array([255/255, 0/255, 0/255, 0.6])  # Red color with alpha
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def mask_extend(img, box, extend_pixels=10, useRectangle=False):
    box[0] = int(box[0])
    box[1] = int(box[1])
    box[2] = int(box[2])
    box[3] = int(box[3])
    region = img.crop(tuple(box))
    new_width = box[2] - box[0] + 2*extend_pixels
    new_height = box[3] - box[1] + 2*extend_pixels

    region_BILINEAR = region.resize((int(new_width), int(new_height)))
    if useRectangle:
        region_draw = ImageDraw.Draw(region_BILINEAR)
        region_draw.rectangle((0, 0, new_width, new_height), fill=(255, 255, 255))    
    
    img.paste(region_BILINEAR, (int(box[0]-extend_pixels), int(box[1]-extend_pixels)))
    return img

def xywh_to_xyxy(box, sizeW, sizeH):
    if isinstance(box, list):
        box = torch.Tensor(box)
    box = box * torch.Tensor([sizeW, sizeH, sizeW, sizeH])
    box[:2] -= box[2:] / 2
    box[2:] += box[:2]
    box = box.numpy()
    return box

def mix_masks(imgs):
    re_img =  1 - np.asarray(imgs[0].convert("1"))
    for i in range(len(imgs)-1):
        re_img = np.multiply(re_img, 1 - np.asarray(imgs[i+1].convert("1")))
    re_img =  1 - re_img
    return  Image.fromarray(np.uint8(255*re_img))

def load_groundingdino_model(device):
    # initialize groundingdino model
    logger.info(f"initialize groundingdino model...")
    groundingdino_model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae, device=device) #'cpu')
    return groundingdino_model

def get_sam_vit_h_4b8939():
    if not os.path.exists('./checkpoints/sam_vit_h_4b8939.pth'):
        logger.info(f"get sam_vit_h_4b8939.pth...")
        result = subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)
        print(f'wget sam_vit_h_4b8939.pth result = {result}')

def load_sam_model(device):
    # initialize SAM
    global sam_model, sam_predictor, sam_mask_generator, sam_device
    get_sam_vit_h_4b8939()
    logger.info(f"initialize SAM model...")
    sam_device = device
    sam_model = build_sam(checkpoint=sam_checkpoint).to(sam_device)
    sam_predictor = SamPredictor(sam_model)
    sam_mask_generator = SamAutomaticMaskGenerator(sam_model)

def load_lama_cleaner_model(device):
    # initialize lama_cleaner
    global lama_cleaner_model
    logger.info(f"initialize lama_cleaner...")

    lama_cleaner_model = ModelManager(
            name='lama',
            device=device,
        )

def lama_cleaner_process(image, mask, cleaner_size_limit=1080):
    try:
        logger.info(f'_______lama_cleaner_process_______') 
        ori_image = image
        if mask.shape[0] == image.shape[1] and mask.shape[1] == image.shape[0] and mask.shape[0] != mask.shape[1]:
            # rotate image

            ori_image = np.transpose(image[::-1, ...][:, ::-1], axes=(1, 0, 2))[::-1, ...]

            image = ori_image
        
        original_shape = ori_image.shape
        interpolation = cv2.INTER_CUBIC
        size_limit = cleaner_size_limit
        if size_limit == -1:
            size_limit = max(image.shape)
        else:
            size_limit = int(size_limit)

        config = lama_Config(
            ldm_steps=25,
            ldm_sampler='plms',
            zits_wireframe=True,
            hd_strategy='Original',
            hd_strategy_crop_margin=196,
            hd_strategy_crop_trigger_size=1280,
            hd_strategy_resize_limit=2048,
            prompt='',
            use_croper=False,
            croper_x=0,
            croper_y=0,
            croper_height=512,
            croper_width=512,
            sd_mask_blur=5,
            sd_strength=0.75,
            sd_steps=50,
            sd_guidance_scale=7.5,
            sd_sampler='ddim',
            sd_seed=42,
            cv2_flag='INPAINT_NS',
            cv2_radius=5,
        )
    
        if config.sd_seed == -1:
            config.sd_seed = random.randint(1, 999999999)

        image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
        # logger.info(f"Resized image shape_1_: {image.shape}")

        mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)
        # logger.info(f"mask image shape_1_: {mask.shape} / {type(mask)}")

        res_np_img = lama_cleaner_model(image, mask, config)

        torch.cuda.empty_cache()
        image = Image.open(io.BytesIO(numpy_to_bytes(res_np_img, 'png')))

    except Exception as e:
        logger.info(f'lama_cleaner_process[Error]:' + str(e))
        image = None        
    return  image

def save_object_image(image, mask):
    # Convert PyTorch tensor to NumPy array
    mask_np = mask.squeeze().cpu().numpy()  # Remove extra dimensions

    # Create a new image with white background
    object_image = Image.new("RGB", image.size, (255, 255, 255))

    # Paste the object region onto the white background based on the mask
    object_region = Image.composite(image, object_image, Image.fromarray((mask_np * 255).astype(np.uint8))).convert('RGBA')
    
    # Save the object region as the specified object image
    # object_region.save(output_path)
    
    return object_region

def dilate_mask(mask, dilate_factor=25):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def run_anything_task(input_image, text_prompt, box_threshold=0.3, text_threshold=0.25, iou_threshold=0.8):

    text_prompt_orig = text_prompt.strip()
    text_prompt = text_prompt_orig.split('_')[0]
    output_images = []
    file_temp = int(time.time())
    
    image_pil, image = load_image(input_image)
    input_img = image_pil

    size = image_pil.size
    H, W = size[1], size[0]
    
    groundingdino_device = 'cuda:0'
        
    boxes_filt, pred_phrases = get_grounding_output(
        groundingdino_model, image, text_prompt, box_threshold, text_threshold, device=groundingdino_device
    )
    
    if boxes_filt.size(0) == 0:
        print('No objects detected, please try others.')
        return []
    
    boxes_filt_ori = copy.deepcopy(boxes_filt)

    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }

    image = np.array(input_img)
    if sam_predictor:
        sam_predictor.set_image(image)

    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.to(sam_device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

    masks, _, _ = sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    # masks: [9, 1, 512, 512]
    assert sam_checkpoint, 'sam_checkpoint is not found!'

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca())
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.cpu().numpy(), plt.gca(), label)
    plt.axis('off')
    
    image_path = os.path.join(output_dir, f"grounding_seg_output_{file_temp}.jpg")
    plt.savefig(image_path, bbox_inches="tight")
    plt.clf()
    plt.close('all')
    
    segment_image_result = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    os.remove(image_path)
    
    output_images.append(Image.fromarray(segment_image_result)) 

    masks_ori = copy.deepcopy(masks)
    mask_fore = torch.sum(masks, dim=0)
    
    masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy()
    mask_pil = Image.fromarray(mask)
    
    image_fore = save_object_image(image_pil, mask_fore)
    output_images.append(image_fore) 
    
    masks_shape = masks_ori.shape    
    boxes_filt_ori_array = boxes_filt_ori.numpy()

    extend_shape_0 = masks_shape[0]
    extend_shape_1 = masks_shape[1]
    mask_imgs = []
    
    for i in range(extend_shape_0):
        for j in range(extend_shape_1):                
            mask = masks_ori[i][j].cpu().numpy()
            mask = dilate_mask(mask)* 255
            mask_pil = Image.fromarray(mask)  
            
            mask_pil_exp = mask_extend(copy.deepcopy(mask_pil).convert("RGB"), 
                            xywh_to_xyxy(torch.tensor(boxes_filt_ori_array[i]), W, H))
            
            mask_imgs.append(mask_pil_exp)
    
    mask_pil = mix_masks(mask_imgs)
    
    image_inpainting = lama_cleaner_process(np.array(image_pil), np.array(mask_pil.convert("L")))

    image_inpainting = image_inpainting.resize((image_pil.size[0], image_pil.size[1]))
    output_images.append(image_inpainting)

    return output_images


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def preprocess_image(input_im):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
    input_im = np.asarray(input_im, dtype=np.float32) / 255.0
    # (H, W, 4) array in [0, 1].

    # apply correct method of compositing to avoid sudden transitions / thresholding
    # (smoothly transition foreground to white background based on alpha values)
    alpha = input_im[:, :, 3:4]
    white_im = np.ones_like(input_im)
    input_im = alpha * input_im + (1.0 - alpha) * white_im

    input_im = input_im[:, :, 0:3]
    # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')

    return input_im


def main_run(models, device,
             x=0.0, y=0.0, z=0.0,
             raw_im=None, scale=3.0, 
             n_samples=4, ddim_steps=50, ddim_eta=1.0, 
             precision='fp32', h=256, w=256):
    '''
    :param raw_im (PIL Image).
    '''
    input_im = preprocess_image(raw_im)
    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(models['turncam'])
    used_x = x
    x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                  ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    return output_ims


def task2(image, azimuth, polar, device_idx=_GPU_INDEX, ckpt='./checkpoints/zero123-xl.ckpt', config='configs/sd-objaverse-finetune-c_concat-256.yaml'):

    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)
    radius = 0.0
    samples=1
    scale = 3
    steps = 100

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)

    gen_output = main_run(models, device, polar, azimuth, radius, image, scale, samples, steps)
    
    return gen_output[0]


def find_object_position(original_image, rotated_object):
    # Convert images to grayscale for template matching
    original_gray = original_image.convert('L')
    rotated_gray = rotated_object.convert('L')

    # Use template matching to find the object in the original image
    result = ImageChops.difference(original_gray, rotated_gray)
    bbox = result.getbbox()

    # Get the top-left corner coordinates of the matched region
    original_object_x, original_object_y = bbox[:2]
    return original_object_x+10, original_object_y+10


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Task 2", add_help=True)
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument("class_name", type=str, help="Text prompt describing the object")
    parser.add_argument("azimuth", type=int, help="azimuth angle")
    parser.add_argument("polar", type=int, help="azimuth angle")
    parser.add_argument("output", type=str, help="Path to save the output image")
    
    args, _ = parser.parse_known_args()
    print(f'args = {args}')
    
    groundingdino_model = load_groundingdino_model('cpu')
    load_sam_model(device)
    load_lama_cleaner_model(device)
    
    output_images = run_anything_task(args.image, args.class_name)
    object_image = output_images[1]
    background_image = output_images[2]
    orig_H, orig_W = background_image.size
    
    rotated_image = task2(object_image, args.azimuth, args.polar)
    rotated_image = rotated_image.resize([orig_H, orig_W], Image.Resampling.LANCZOS)
    object_image = object_image.resize([orig_H, orig_W], Image.Resampling.LANCZOS)
    
    # Get the coordinates of the object in the original image using template matching
    original_object_x, original_object_y = find_object_position(object_image, rotated_image)
    
    # Create a mask based on non-white pixels in the rotated object
    mask = rotated_image.convert("L").point(lambda pixel: 255 if pixel < 245 else 0)
    
    # Perform erosion on the mask
    mask = mask.filter(ImageFilter.MinFilter(size=5))

    # Create a new image with an alpha channel
    rotated_object_with_alpha = Image.new("RGBA", rotated_image.size, (0, 0, 0, 0))

    # Paste the RGB channels of the rotated object onto the new image
    rotated_object_with_alpha.paste(rotated_image, (0, 0), mask)

    # Paste the rotated object with alpha onto the background at the same position
    background_image.paste(rotated_object_with_alpha, (original_object_x, original_object_y), rotated_object_with_alpha)
    
    image_path = os.path.join(output_dir, args.output)
    background_image.save(image_path)
    
    image_path = os.path.join(output_dir, f'generated_mask_{"_".join(args.output.split("_")[1:]).replace(".png", "")}.png')
    output_images[0].save(image_path)  