"""SAMPLING ONLY."""
from einops import rearrange, repeat
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import partial
from ldm.modules.attention import get_global_heat_map, clear_heat_maps, get_rank,edit_rank,clear_rank
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from ldm.modules.attention import get_global_heat_map, clear_heat_maps,next_heat_map
from copy import deepcopy
from matplotlib import pyplot as plt
import os
import time

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        # TODO 这里可以加mask？
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates



    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        # TODO 这里加mask?
        for i, step in enumerate(iterator):

            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            print("time", i, step,ts)

            if mask is not None:# and (i< (total_steps-5) ): #TODO 好家伙，一开，完全和原图一样
                # assert x0 is not None #TODO 确定性前向？目前没加噪声，应该用Unet的输出做高斯噪声
                img_orig = self.model.q_sample_(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            #TODO 不在img加mask，在attention的某一维
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            next_heat_map()


            # mask = get_global_heat_map().mean(0)#TODO 选了所有词的mean()，也许要改


            # if i<5:
            # # if mask==None:
            #     mask=get_global_heat_map()[:8].mean(0)/(get_global_heat_map()[:8].max()-get_global_heat_map()[:8].min())
            # # else:
            # #     mask=0.5*mask+0.5*(get_global_heat_map()[:6].mean(0)/(get_global_heat_map()[:6].max()-get_global_heat_map()[:6].min()))
            #     # 可不可以soft一点？
            #     mask[mask <= mask.mean()] = 0
            #     mask[mask > mask.mean()] = 1
            #     #
            #     mask=mask.to(img.device)
            #     if torch.isnan(mask).any():
            #         mask=None

            # else:
            #     mask = None
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,edit_con=None,edit_guidance_scale=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        elif edit_con is not None:
            x_in = torch.cat([x] * 3)
            t_in = torch.cat([t] * 3)
            c_in = torch.cat([unconditional_conditioning,edit_con, c])
            result=self.model.apply_model(x_in, t_in, c_in)
            if result.shape[0]==2:
                e_t_uncond, e_t = result.chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            elif result.shape[0]==3:
                e_t_uncond,e_t_edit, e_t = result.chunk(3)
                edit_beta=edit_guidance_scale#0.95
                e_t_mid=edit_beta*e_t+(1-edit_beta)*e_t_edit # edit_beta*e_t+
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t_mid - e_t_uncond)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):#TODO 加了unconditional 变成（2,77,768）
                        c_in[k] = [
                            torch.cat([unconditional_conditioning[k][i], c[k][i]])
                            for i in range(len(c[k]))
                        ]
                    else:
                        c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)#(1,4,64,64)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0 #(1,4,64,64)

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,x0=None,
               use_original_steps=False,edit_con=None,edit_guidance_scale=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        rank = get_rank()
        mask1 = None  # 可以固定为最大的那个
        noise=None #固定了每轮的noise mask
        # x0=None
        if rank != {} and ('mask' in list(rank.keys())) :#and False:  # and not os.path.exists("./mask1.png"):
            # print(rank)
            ratio = rank['mask'][0]
            mask = rank['mask'][1]

            mask_min = mask.min()
            mask_max = mask.max()
            noise = torch.rand(mask.shape[0], mask.shape[1], device=mask.device)
            mask_norm = (mask - mask_min) / (mask_max - mask_min)


            mask1=torch.zeros_like(mask)
            # mask1[mask_norm <= mask_norm.mean()] = 0 #还是不动比较好
            mask1[mask_norm > mask_norm.mean()] = 255
            # 0是黑，要修改的；255是白，要弱保护的
            # 也许应该可视化一下

            # mask1 = mask1 * noise
            # m1 = mask1.lt(1-ratio)#包括0和一些
            # m2 = mask1.gt(ratio)  # 大于ratio的概率为1 ？，
            # mask1=mask1.masked_fill(m1, 0)
            # mask1=mask1.masked_fill(~m1, 255)

            # if True:#not os.path.exists("./mask1.png"):
            mask_view = torch.concat((mask1.unsqueeze(2), mask1.unsqueeze(2), mask1.unsqueeze(2)), 2)  # 4个通道
            plt.axis('off')  # 去坐标轴
            plt.xticks([])  # 去 x 轴刻度
            plt.yticks([])  # 去 y 轴刻度
            plt.imshow(mask_view.cpu().detach().numpy().astype(np.uint8))
            plt.savefig("./mask1.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            mask1[mask1[:, :] == 255] = 1

        x_dec = deepcopy(x_latent)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            if mask1 is not None: # ddpm搜mask
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                #TODO mask随机，且ratio不变
                mask11=deepcopy(mask1)
                noise = torch.rand(mask11.shape[0], mask11.shape[1], device=mask11.device)
                mask11 = mask11 * noise
                m1 = mask11.lt(1 - ratio)#max((1 - ratio),(1-float(1.0*index/total_steps))))  # 小于0.8的 包括0和一些
                mask11 = mask11.masked_fill(m1, 0)
                mask11 = mask11.masked_fill(~m1, 1)#1)
                x_dec = img_orig * mask11 + (1. - mask11) * x_dec

                #TODO mask_ratio减小
                #min(float(1.0*index/total_steps) ,ratio) #1-ratio比1效果好，越小越靠x_dec

                # noise = torch.rand(mask1.shape[0], mask1.shape[1], device=mask1.device)
                # mask1 = mask1 * noise
                # m1 = mask1.lt((1 - ratio))  # 小于0.8的 包括0和一些
                # mask1 = mask1.masked_fill(m1, 0)
                # mask1 = mask1.masked_fill(~m1, 1)
                # x_dec = img_orig * mask1 + (1. - mask1) * x_dec

                # 可视化mask1
                tic = time.time()
                mask3 = deepcopy(mask11)
                mask3[mask3[:,:]==1]=255
                mask_view = torch.concat((mask3.unsqueeze(2), mask3.unsqueeze(2), mask3.unsqueeze(2)), 2)  # 4个通道
                plt.axis('off')  # 去坐标轴
                plt.xticks([])  # 去 x 轴刻度
                plt.yticks([])  # 去 y 轴刻度
                plt.imshow(mask_view.cpu().detach().numpy().astype(np.uint8))
                mask_path="./mask/"
                os.makedirs(mask_path, exist_ok=True)
                plt.savefig(mask_path+str(tic)+".png", bbox_inches='tight', pad_inches=0)
                plt.close()


            #x_dec已根据epis恢复成z_0
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,edit_con=edit_con,edit_guidance_scale=edit_guidance_scale)
            next_heat_map()
            x_samples =  self.model.decode_first_stage(x_dec)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
            Image.fromarray(x_sample.astype(np.uint8)).save("./{base_count:05}.png")
            # 可视化一下latent space

        return x_dec
