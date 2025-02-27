# This file extracts CTC cells from populated scenes
import glob
from skimage import io
import numpy as np
from argparse import ArgumentParser
from pathlib import Path


def set_data(args):
    imgs_path = args.data_root + args.cell_group
    st_path = args.data_root + args.cell_group + "_ST/SEG"

    imgs = []
    imgs.append(
        [
            io.imread(img, plugin="pil")
            for img in sorted(glob.glob(imgs_path + "/*.tif"))
        ]
    )
    num_frames = len(imgs[0])
    imgs = np.stack(imgs[0])
    masks = []
    masks.append(
        [
            io.imread(mask, plugin="pil")
            for mask in sorted(glob.glob(st_path + "/*.tif"))
        ]
    )
    masks = np.stack(masks[0])
    labels = np.unique(masks)

    # here, we normalize the mask values for that accordingly
    max_val = args.max_val
    min_val = args.min_val 

    # ignore the label 0 because it is the background
    for label in labels[1:]:
        Path(args.new_data_folder + str(label).zfill(3)).mkdir(
            parents=True, exist_ok=True
        )
        Path(args.new_mask_folder + str(label).zfill(3)).mkdir(
            parents=True, exist_ok=True
        )

        for frame in range(num_frames):

            empty_img = (np.ones((128, 128)) * min_val).astype(np.float32)
            empty_mask = np.zeros((128, 128)).astype(np.float32)
            indices = np.where(masks[frame] == label)
            if len(indices[0]) == 0:
                continue
            
            empty_img[
                indices[0]
                + 64
                - int(0.5 * np.min(indices[0]) + 0.5 * np.max(indices[0])),
                indices[1]
                + 64
                - int(0.5 * np.min(indices[1]) + 0.5 * np.max(indices[1])),
            ] = imgs[frame][indices]
            # average the image foreground values to assign it to mask foreground
            foreground = np.average(
                (imgs[frame][indices] - min_val) / (max_val - min_val)
            )
            empty_mask[
                indices[0]
                + 64
                - int(0.5 * np.min(indices[0]) + 0.5 * np.max(indices[0])),
                indices[1]
                + 64
                - int(0.5 * np.min(indices[1]) + 0.5 * np.max(indices[1])),
            ] = foreground

            io.imsave(
                args.new_data_folder
                + str(label).zfill(3)
                + "/"
                + str(frame).zfill(2)
                + ".tif",
                empty_img.astype(np.uint16),
            )
            io.imsave(
                args.new_mask_folder
                + str(label).zfill(3)
                + "/"
                + str(frame).zfill(2)
                + ".tif",
                empty_mask,
            )


def get_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="path for the input images",
    )

    parser.add_argument(
        "--new_data_folder",
        type=str,
        required=True,
        help="path for the output images",
    )

    parser.add_argument(
        "--new_mask_folder",
        type=str,
        required=True,
        help="path for the output masks",
    )

    parser.add_argument(
        "--cell_group",
        type=str,
        default="01",
        help="image group to be selected from the dataset",
    )

    parser.add_argument(
        "--max_val",
        type=int,
        default=36863,
        help="maximum pixel value",
    )

    parser.add_argument(
        "--min_val",
        type=int,
        default=32995,
        help="minimum pixel value",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    set_data(args)
