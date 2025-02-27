from share import *
import config

from skimage import io
import einops
import numpy as np
import torch
import glob
from pytorch_lightning import seed_everything
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from skimage import io
import torch.nn.functional as F
import argparse

apply_uniformer = UniformerDetector()
# Seeding makes everything deterministic
seed_everything(1)

def process(mask_filename, prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, eta, ddim_sampler, model):
    with torch.no_grad():
        # Min and max values for images in this dataset
        mask = io.imread(mask_filename)
        # invert the syncdreamer settings
        mask[mask==255] = 0
        # Normalize to [0,1] 
        mask = mask / np.max(mask)
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        mask = torch.from_numpy(mask.astype(np.float32))
        mask = torch.squeeze(F.interpolate(mask[None,None,...], [mask.shape[0]*2, mask.shape[1]*2], mode='nearest'))
        input_image = torch.stack([mask] * 3, dim=-1)[:512, :512, :].cuda()
        H, W, C = input_image.shape
        input_image = einops.rearrange(input_image[None,...], 'b h w c -> b c h w').clone()
        cond = {"c_concat": [input_image], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [input_image], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = samples
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255)

        results = [x_samples[i] for i in range(num_samples)]
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for cell generation')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--masks', type=str, required=True, help='Path to the pregenerated conditional masks')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save the outputs')
    parser.add_argument('--prompt', type=str, default="", help='Text prompt for generation')
    parser.add_argument('--n_prompt', type=str, default='', help='Negative text prompt for generation')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM steps')
    parser.add_argument('--guess_mode', type=bool, default=True, help='Guess mode for generation')
    parser.add_argument('--strength', type=float, default=2.0, help='Control strength for the text prompt')
    parser.add_argument('--scale', type=float, default=5.0, help='Guidance scale for classifier-free guidance')
    parser.add_argument('--eta', type=float, default=0.0, help='Eta value for DDIM')

    return parser.parse_args()

def infer(args):
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(args.pretrained_model, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)


    prompt = args.prompt
    n_prompt = args.n_prompt
    num_samples = args.num_samples
    ddim_steps = args.ddim_steps
    guess_mode = args.guess_mode
    strength = args.strength
    scale = args.scale
    eta = args.eta

    masks = sorted(glob.glob(args.masks + "/*"))
    for i,mask in enumerate(masks):
        output = process(mask, prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, eta, ddim_sampler, model)
        io.imsave(args.out_path + str(i).zfill(4)+'.tif', np.stack(output)[...,0])


if __name__ == "__main__":
    args = parse_args()
    infer(args)