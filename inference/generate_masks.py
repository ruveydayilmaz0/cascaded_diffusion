# Adapted from https://github.com/huggingface/diffusers

from diffusers import DDIMPipeline
import numpy as np
from skimage import io
import argparse


# Configs
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train cells with cascaded diffusion model."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the outputs."
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Path to the pretrained model folder.",
    )
    parser.add_argument(
        "--num_masks", type=int, required=True, help="Number of masks to generate."
    )
    parser.add_argument(
        "--diff_steps",
        type=int,
        default=50,
        help="Number of diffusion timesteps to take.",
    )

    return parser.parse_args()


def generate_masks(args):

    # load model and scheduler
    ddim = DDIMPipeline.from_pretrained(args.pretrained_model)

    for i in range(args.num_masks):
        image = ddim(num_inference_steps=args.diff_steps, eta=1).images[0]
        image = np.array(image)
        image[image != 0] = 255

        # save the image
        io.imsave(args.output_path + str(i).zfill(4) + ".png", image.astype(np.uint8))


if __name__ == "__main__":
    args = parse_args()
    generate_masks(args)
