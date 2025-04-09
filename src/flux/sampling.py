import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        
        img = img + (t_prev - t_curr) * pred

    return img, info


def denoise_rf_solver(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info


def denoise_fireflow(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    next_step_velocity = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        if next_step_velocity is None:
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info
            )
        else:
            pred = next_step_velocity
        
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )
        next_step_velocity = pred_mid
        
        img = img + (t_prev - t_curr) * pred_mid

    return img, info


def denoise_midpoint(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )
        next_step_velocity = pred_mid
        
        img = img + (t_prev - t_curr) * pred_mid

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def denoise_adaptive(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0,
    acceleration_threshold: float = 5.0,  # 曲率变化阈值
    min_step_ratio: float = 0.5        # 最小步长比例
):
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])
    
    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
    # 转换为可修改的列表
    remaining_steps = list(zip(timesteps[:-1], timesteps[1:]))

    info['forward_steps'] = info.get('forward_steps', 0)
    
    while remaining_steps:
        t_curr, t_prev = remaining_steps.pop(0)
        delta_t = t_prev - t_curr
        
        # 第一次预测
        t_vec = torch.full_like(img[:,0,0], t_curr)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[len(timesteps)-len(remaining_steps)-2]
        
        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        info['forward_steps'] += 1
        
        # 中间点预测
        img_mid = img + delta_t/2 * pred
        t_mid = t_curr + delta_t/2
        
        # 第二次预测（用于曲率计算）
        pred_mid, _ = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=torch.full_like(img[:,0,0], t_mid),
            guidance=guidance_vec,
            info=info
        )
        info['forward_steps'] += 1
        
        # 计算曲率变化
        acceleration = torch.mean(torch.abs(pred_mid - pred) / (delta_t/2 + 1e-6))
        step_ratio = 1.0 if acceleration < acceleration_threshold else min_step_ratio
        
        # 调整实际使用的步长
        actual_delta_t = delta_t * step_ratio
        if step_ratio < 1.0:
            # 插入新的时间步
            new_t = t_curr + actual_delta_t
            remaining_steps.insert(0, (new_t, t_prev))
        
        # 最终更新
        img = img + actual_delta_t * pred + 0.5 * (actual_delta_t**2) * (pred_mid - pred)/ (delta_t/2)
    
    return img, info

def denoist_Ralston(
        model: Flux,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        timesteps: list[float],
        inverse,
        info,
        guidance: float = 4.0
):
    """
    Ralston Method:

    k1 = h * f(t_i, y_i)
    k2 = h * f(t_i + (2/3)h, y_i + (2/3)k1)
    y_{i+1} = y_i + (1/3)k1 + (2/3)k2

    """
    # this is ignored for schnell, only inject at the first serval steps
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['inject'] = inject_list[i]
        info['second_order'] = False

        pred, info = model(
            img=img,
            img_ids = img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        k1 = pred * (t_prev - t_curr)

        img_mid = img + (2/3) * k1
        t_vec_mid = torch.full((img.shape[0],), (t_curr + 2/3 *(t_prev - t_curr)), dtype=img.dtype, device=img.device)
        info['second_order'] = True 

        pred_mid, info = model(
            img = img_mid,
            img_ids = img_ids,
            txt = txt,
            txt_ids = txt_ids,
            y = vec,
            timesteps = t_vec_mid,
            guidance = guidance_vec,
            info = info
            )

        k2 = (t_prev - t_curr) * pred_mid

        img = img + 1/3 * k1 + 2/3 * k2

    return img, info 

def denoise_diffusion(
        model: Flux,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        timesteps: list[float],
        inverse,
        info,
        guidance: float = 4.0
):
    """
    dX_t = (u^{theta}_t(x) + 1/2 sigma^2 score_t^{theta}(x))*dt + sigma*dW_t

    p_t(x|z) = N(x; alpha_t z, beta_t^2 I_d)

    alpha_t =  t, beta_t^2 = 1 - t --> x = t * z + sqrt(1 - t) epsilon --> epsilon ~ N(0, Id)

    score_t(x) = (alpha_t * u_t (x) - d(alpha_t) * x) / (beta_t^2 * d(alpha_t) - alpha_t * d(beta_t) * beta_t)
    """

    # this is ignored for schnell, only inject at the first serval steps
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['inject'] = inject_list[i]
        info['second_order'] = False

        # u^{theta}_t(x)
        pred, info = model(
            img=img,
            img_ids = img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        #transform into score
        score = (t_curr * pred - img) / (1- t_curr / 2)

        #following the SDE equation
        sigma = 0.001
        GaussianNoise = torch.randn_like(img) # sigma dW_t = sigma * sqrt(t - s) * epsilon
        
        img = img + (pred + 0.5 * sigma**2 * score)*(t_prev - t_curr) + sigma * torch.sqrt(torch.abs(torch.tensor(t_prev - t_curr))) * GaussianNoise

    return img, info 

def denoise_diffusion_sigma_timedependent(
        model: Flux,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        timesteps: list[float],
        inverse,
        info,
        guidance: float = 4.0
):
    """
    sigma = 0.001 * e^{-5t}
    dX_t = (u^{theta}_t(x) + 1/2 sigma^2 score_t^{theta}(x))*dt + sigma*dW_t

    p_t(x|z) = N(x; alpha_t z, beta_t^2 I_d)

    alpha_t =  t, beta_t^2 = 1 - t --> x = t * z + sqrt(1 - t) epsilon --> epsilon ~ N(0, Id)

    score_t(x) = (alpha_t * u_t (x) - d(alpha_t) * x) / (beta_t^2 * d(alpha_t) - alpha_t * d(beta_t) * beta_t)
    """

    # this is ignored for schnell, only inject at the first serval steps
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['inject'] = inject_list[i]
        info['second_order'] = False

        # u^{theta}_t(x)
        pred, info = model(
            img=img,
            img_ids = img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        #transform into score
        score = (t_curr * pred - img) / (1- t_curr / 2)

        #following the SDE equation
        sigma = 0.001 * torch.exp(-5 * torch.tensor(t_curr))
        GaussianNoise = torch.randn_like(img) # sigma dW_t = sigma * sqrt(t - s) * epsilon
        
        img = img + (pred + 0.5 * sigma**2 * score)*(t_prev - t_curr) + sigma * torch.sqrt(torch.abs(torch.tensor(t_prev - t_curr))) * GaussianNoise

    return img, info

def denoise_diffusion_free_noise(
        model: Flux,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        timesteps: list[float],
        inverse,
        info,
        guidance: float = 4.0
):
    """
    sigma = 0.001 * e^{-5t}
    dX_t = (u^{theta}_t(x) + 1/2 sigma^2 score_t^{theta}(x))*dt + sigma*dW_t

    p_t(x|z) = N(x; alpha_t z, beta_t^2 I_d)

    alpha_t =  t, beta_t^2 = 1 - t --> x = t * z + sqrt(1 - t) epsilon --> epsilon ~ N(0, Id)

    score_t(x) = (alpha_t * u_t (x) - d(alpha_t) * x) / (beta_t^2 * d(alpha_t) - alpha_t * d(beta_t) * beta_t)
    """

    # this is ignored for schnell, only inject at the first serval steps
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['inject'] = inject_list[i]
        info['second_order'] = False

        # u^{theta}_t(x)
        pred, info = model(
            img=img,
            img_ids = img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        #transform into score
        score = (t_curr * pred - img) / (1- t_curr / 2)

        #following the SDE equation
        sigma = 0.001 * torch.exp(-5 * torch.tensor(t_curr))
        GaussianNoise = torch.randn_like(img) # sigma dW_t = sigma * sqrt(t - s) * epsilon
        
        img = img + (pred + 0.5 * sigma**2 * score)*(t_prev - t_curr)

    return img, info