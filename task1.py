import warnings
warnings.filterwarnings('ignore')

import subprocess, io, os, sys, time
import gradio as gr
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, './GroundingDINO')

import argparse
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download
from lama_cleaner.helper import (
    load_img,
    numpy_to_bytes,
    resize_max_size,
)

import ast

config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
sam_checkpoint = './checkpoints/sam_vit_h_4b8939.pth' 
output_dir = "outputs_segment"

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

def save_object_image(image, mask, output_path):
    # Convert PyTorch tensor to NumPy array
    mask_np = mask.squeeze().cpu().numpy()  # Remove extra dimensions

    # Create a new image with white background
    object_image = Image.new("RGB", image.size, (255, 255, 255))

    # Paste the object region onto the white background based on the mask
    object_region = Image.composite(image, object_image, Image.fromarray((mask_np * 255).astype(np.uint8))).convert('RGBA')
    
    # Save the object region as the specified object image
    object_region.save(output_path)
    
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
    
    object_output_path = os.path.join(output_dir, f"object_{text_prompt_orig}.png")
    image_fore = save_object_image(image_pil, mask_fore, object_output_path)
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


def task1(args):
    output_images = run_anything_task(args.input_image, args.class_name)
    
    for i, image in enumerate(output_images):
        if i == 0:
            image_path = os.path.join(output_dir, args.output)
            image.save(image_path)
        elif i == 2:
            image_path = os.path.join(output_dir, f"background_image_{args.class_name}.png")
            image.save(image_path)
        else:
            pass
                             
    print(f'DONE')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task 1", add_help=True)
    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument("class_name", type=str, help="Text prompt describing the object")
    parser.add_argument("output", type=str, help="Path to save the output image")

    args, _ = parser.parse_known_args()
    print(f'args = {args}')

    groundingdino_model = load_groundingdino_model('cpu')
    load_sam_model(device)
    load_lama_cleaner_model(device)
    
    task1(args)