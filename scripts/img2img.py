"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from matplotlib import pyplot as plt
from copy import deepcopy
from ldm.modules.attention import get_global_heat_map, clear_heat_maps, get_rank,edit_rank,clear_rank
import torch.nn.functional as F
from collections import defaultdict

# rank=defaultdict(list)
#
# def get_rank():
#     global rank
#     return rank

def expand_m(m, n: int = 1, o=512, mode='bicubic'):
    m = m.unsqueeze(0).unsqueeze(0) / n
    m = F.interpolate(m.float().detach(), size=(o, o), mode='bicubic', align_corners=False)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = m.cpu().detach()

    return m


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()
    load_ori="/home/wenyi_mo/stable-diffusion-main/models/ldm/stable-diffusion-v1/model.ckpt"
    load_9_bird = "/home/wenyi_mo/stable-diffusion/finetune_logs/2022-11-08T18-07-52_one_pic/checkpoints/last.ckpt"
    load_1_bird = "/home/wenyi_mo/stable-diffusion/finetune_logs/2022-11-08T18-07-52_one_pic/checkpoints/epoch=000001.ckpt"
    load_7_bird="/home/wenyi_mo/stable-diffusion/ok/bird/epoch=000007.ckpt"
    parser.add_argument(
        "--ckpt",
        type=str,
        default=load_ori,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A photo of a bird spreading wings.",#"A photo of a bird wearing hat",##"A picture of a bird, Monet style.",
        help="the prompt to render"
    )
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="../outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,#TODO 可加大
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/latent-diffusion/one_pic.yaml",#"configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    #TODO lamb
    lamb=0.7


    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        #TODO get attention map
                        t_enc_begin = int(2)#TODO 改为2
                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc_begin]*batch_size).to(device))
                        #TODO 清空heat map
                        clear_heat_maps()
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc_begin, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        heat_maps = get_global_heat_map()
                        #[77, 64, 64]#heat_maps[0].max()124.3595,min 113.3953 其他idx在0.2~1.8之间
                        for num in range(len(prompt.split(' '))+1):
                            mplot = expand_m(heat_maps[num], 1)

                            mask = torch.zeros_like(mplot.squeeze(0))
                            mask[mplot.squeeze(0) < 0.5 * mplot.squeeze(0).max()] = 1

                            mask1 = torch.concat((mask, mask, mask), 0) #三个通道
                            x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
                            x_sample1=deepcopy(x_sample)
                            x_sample1[:, :, :][mask1.permute(1, 2, 0)[:, :, :] > 0] = 255
                            str_path = "../outputs/img2img-samples1/mask_before"
                            if not os.path.exists(str_path):
                                os.makedirs(str_path)
                            if num==0:
                                str_path1 = str_path + "/soft_start.png"
                            else:
                                str_path1= str_path+"/soft_" + prompt.split(' ')[num - 1] + ".png"

                            plt.imshow(x_sample1.astype(np.uint8))
                            plt.savefig(str_path1)
                            plt.close()

                        #TODO 也需要另外保存heat_maps

                        for i in range((len(prompt.split(' '))+2)):
                            if i==0:
                                print(heat_maps[i].sum(),"start")
                            elif i>len(prompt.split(' ')):
                                print(heat_maps[i].sum(),"end")
                            else:
                                print(heat_maps[i].sum(),prompt.split(' ')[i-1])


                        b=[(-1)*float(heat_maps[i].sum()) for i in range(1,len(prompt.split(' '))+1)]#b去掉了start token
                        b_sum=np.sum(b)
                        # r_b=[i / b_sum for i in b] #用比率
                        # TODO 选出前50%的txt emb
                        clear_rank()
                        rank = defaultdict(list)
                        emp_str=""
                        for i in (np.argsort(b)[int(len(b)*0.5):]):
                            print(prompt.split(' ')[i])
                            if emp_str=="":
                                emp_str=prompt.split(' ')[i]
                            else:
                                emp_str=emp_str+' '+prompt.split(' ')[i]
                            rank[i+1]=[np.exp(-b[i]/b_sum),heat_maps[i+1]]#+1是start token

                        if emp_str!="":
                            print("emp_str",emp_str)
                            ec = model.get_learned_conditioning(batch_size * [emp_str])
                            uc = lamb *uc + (1-lamb)*ec

                        edit_rank(rank)
                        # rank=np.argsort(b)[int(len(b)*0.5):]+1 #+1是start token
                        # rank {77中的下标：权重}
                        c_new = deepcopy(c)
                        for i in range(len(rank)):
                            c_new[:,int(list(rank.keys())[i]),:]=c_new[:,0,:]#换为start token对应的emb

                        # TODO 清空heat_maps
                        clear_heat_maps()


                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c_new, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)



                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples)

                        # TODO draw attention map
                        heat_maps = get_global_heat_map()
                        for num in range(len(prompt.split(' '))+1):
                            mplot = expand_m(heat_maps[num], 1)

                            mask = torch.zeros_like(mplot.squeeze(0))
                            mask[mplot.squeeze(0) < 0.5 * mplot.squeeze(0).max()] = 1

                            mask1 = torch.concat((mask, mask, mask), 0) #三个通道

                            x_sample1=deepcopy(x_sample)
                            x_sample1[:, :, :][mask1.permute(1, 2, 0)[:, :, :] > 0] = 255
                            str_path = "../outputs/img2img-samples1/mask"
                            if not os.path.exists(str_path):
                                os.makedirs(str_path)
                            if num==0:
                                str_path1 = str_path + "/soft_start.png"
                            else:
                                str_path1= str_path+"/soft_" + prompt.split(' ')[num - 1] + ".png"

                            plt.imshow(x_sample1.astype(np.uint8))
                            plt.savefig(str_path1)
                            plt.close()


                outpath2 = outpath + "/" + str(grid_count) + "scl" + str(opt.scale) + "sten" + str(
                    opt.strength) + "_" + str(opt.prompt)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    outpath1 = outpath2 + ".png"
                    Image.fromarray(grid.astype(np.uint8)).save(outpath1)
                    # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1



    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
