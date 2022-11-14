
import streamlit as st
from einops import repeat
# from streamlit_drawable_canvas import st_canvas
import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import time
import os
from torchvision.utils import make_grid
import math
from ldm.modules.attention import get_global_heat_map, clear_heat_maps
import torch.nn.functional as F
from einops import rearrange, repeat
from matplotlib import pyplot as plt

MAX_SIZE = 640

# load safety model
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from imwatermark import WatermarkEncoder
import cv2

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
wm = "StableDiffusionV1-Inpainting"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))


def get_concat_h_multi_resize(im_list, n_rows, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)

    im_list_resize = [
        im.resize((int(im.width * min_height / im.height), min_height), resample=resample) for im in im_list
    ]

    max_width = max(im.width for im in im_list_resize)

    n_cols = math.ceil(len(im_list) / n_rows)
    total_width = (max_width) * n_cols  # compute total width
    total_height = (min_height) * n_rows

    dst = Image.new("RGB", (total_width, total_height))  # calculate the total height

    pos_x = 0
    pos_y = 0
    for i in range(len(im_list_resize)):
        im = im_list_resize[i]
        dst.paste(im, (pos_y, pos_x))

        pos_y += im.width

        if (i + 1) % n_cols == 0:
            # new row
            pos_x += im.height
            pos_y = 0

    # print(" total: ", total_width, total_height)
    return dst


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def put_watermark(img):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    return x_checked_image, has_nsfw_concept


def expand_m(m, n: int = 1, o=512, mode='bicubic'):
    m = m.unsqueeze(0).unsqueeze(0) / n
    m = F.interpolate(m.float().detach(), size=(o, o), mode='bicubic', align_corners=False)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = m.cpu().detach()

    return m

@st.cache(allow_output_mutation=True)
def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    # model.get_first_stage_encoding(model.encode_first_stage(cc))
    image0 = np.array(image.convert("RGB"))
    image0 = image0[None].transpose(0, 3, 1, 2)
    image0 = torch.from_numpy(image0).to(dtype=torch.float32) / 127.5 - 1.0
    x0=model.get_first_stage_encoding(model.encode_first_stage(repeat(image0.to(device=device), "1 ... -> n ...", n=num_samples)))



    # start_code = x0
    start_code=prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)
    # TODO start_code也许可以不是噪声. 维度是[1,4,64,64]
    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()#[1,5,64,64],5==1+4
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:# mask下采样到[1,1,64,64]，与VAE出来大小一样
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:#[1,4,64,64]
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))#mask_img放进VAE
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]
            samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
                    x0=x0,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)
            result, has_nsfw_concept = check_safety(result)
            result = result*255

    result1 = [Image.fromarray(img.astype(np.uint8)) for img in result]
    result1 = [put_watermark(img) for img in result1]
    return result,result1


if __name__ == "__main__":
    # st.title("Stable Diffusion Inpainting")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a bird wearing a hat",#"A lion runs in the grassland",#
        help="the prompt to render"
    )
    opt = parser.parse_args()
    load_path="../configs/stable-diffusion/v1-inpainting-inference.yaml"
    # ckpt_path="/home/wenyi_mo/stable-diffusion-main/models/ldm/inpainting_big/last.ckpt"
    ckpt_path="/home/wenyi_mo/stable-diffusion-main/models/ldm/inpainting_big/sd-v1-5-inpainting.ckpt"
    sampler = initialize_model(load_path, ckpt_path)

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    image = [x.replace("_mask.png", ".png") for x in masks]
    if image:
        image = Image.open(image[0]).convert('RGB')
        image_source=image
        w, h = image.size
        # print(f"loaded input image of size ({w}, {h})")
        if max(w, h) > MAX_SIZE:
            factor = MAX_SIZE / max(w, h)
            w = int(factor*w)
            h = int(factor*h)
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))
        # print(f"resized to ({width}, {height})")

        prompt = opt.prompt

        seed = 42
        num_samples = 1
        scale = 5 #不能设太大，否则会全黑
        ddim_steps = 10 #30

        fill_color = "rgba(255, 255, 255, 0.0)"
        stroke_width = 64
        stroke_color = "rgba(255, 255, 255, 1.0)"
        bg_color = "rgba(0, 0, 0, 1.0)"
        # drawing_mode = "freedraw"

        masks = Image.open(masks[0]).convert('RGB')
        masks=np.array(masks).astype(np.uint8)
        mask = masks[:, :, -1] > 0
        clear_heat_maps()

        if mask.sum() > 0:
            mask = Image.fromarray(mask)

            result_tensor,result = inpaint(
                sampler=sampler,
                image=image,
                mask=mask,
                prompt=prompt,
                seed=seed,
                scale=scale,
                ddim_steps=ddim_steps,
                num_samples=num_samples,
                h=height, w=width
            )
            # st.write("Inpainted")
            out_dir="../outputs/inpaint/"

            now = int(time.time())
            timeArr = time.localtime(now)
            other_StyleTime = time.strftime("%m-%d/%H:%M", timeArr)
            out_dir=out_dir+other_StyleTime+str(prompt)+"_scale"+str(scale) +'/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            i=0
            for image in result:
                pic_dir=out_dir+str(i)+".png"
                # image.save(pic_dir)
                # mask_dir = out_dir + str(i) + "_mask.png"
                # mask.save(mask_dir)
                get_concat_h_multi_resize(im_list=[image_source, mask,image], n_rows=1).save(pic_dir)
                i=i+1

        heat_maps = get_global_heat_map()#(idx=0)
        #get_global_heat_map(idx=0).shape #TODO 可获得不同时间步的热力图
        for num in range(len(prompt.split(' '))):

            mplot = expand_m(heat_maps[num], 1)
            mask1 = torch.ones_like(mplot)
            mask1[mplot < 0.5 * mplot.max()] = 0
            spotlit_im = torch.tensor(result_tensor[0]).cpu().float().detach()
            # TODO Image.fromarray(result_tensor[0].astype(np.uint8)).save("./b.png")
            # Image.fromarray(spotlit_im.numpy().astype(np.uint8)).save("./b1.png")

            # spotlit_im2 = torch.cat((spotlit_im, (1 - mplot.squeeze(0)).pow(1)).permute(1, 2, 0), dim=3)
            spotlit_im2 = torch.cat((spotlit_im.permute(2, 0, 1), (mplot.squeeze(0)).pow(1)), dim=0)
            # a=spotlit_im.permute(0, 3, 1, 2) * mask1.squeeze(0)
            #
            # x_sample = 255. * rearrange(a.squeeze(0).cpu().numpy(), 'c h w -> h w c')
            # Image.fromarray(x_sample.astype(np.uint8)).save("./b0.png")

            # Image.fromarray(spotlit_im2.permute(1, 2, 0).numpy().astype(np.uint8)).save("./b1.png")

            fig, ax = plt.subplots()

            # ax.imshow(mplot.squeeze().numpy(), cmap='jet')
            ax.imshow(result_tensor[0].astype(np.uint8))
            ax.imshow(spotlit_im2.permute(1, 2, 0).numpy())
            str_path=out_dir+"soft_"+prompt.split(' ')[num - 1]+".png"
            plt.savefig(str_path)

            # im2=spotlit_im.permute(2, 0, 1) * mask1.squeeze(0)
            # ax.imshow(im2.permute(1, 2, 0).numpy())
            # str_path =out_dir +"hard_"+ prompt.split(' ')[num - 1] + ".png"
            # plt.savefig(str_path)
        # print("enjoy")


