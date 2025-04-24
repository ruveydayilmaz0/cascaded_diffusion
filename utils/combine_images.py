import glob
import random
import numpy as np
from skimage import io
from PIL import Image
from utils.cell_placement import CellPlacement
from random import randrange
import os
from scipy import ndimage
import cv2

num_imgs = 400
mask_paths = #PATH_TO_MASKS
image_paths = #PATH_TO_IMAGES
save_path = #PATH_TO_SAVE
settings = {}
settings["imageSize"] = (512, 708)
settings["borderPadding"] = 0
synthetic_image_folders = sorted(glob.glob(image_paths+"*"))
synthetic_mask_folders = sorted(glob.glob(mask_paths+"*"))
patch = np.zeros((20,20))
# def custom_filter(image):
#     return np.amax(image) - np.amin(image)

for j in range(num_imgs):
    num = randrange(40, 50)
    rotations = []
    print("Generating image ", str(j))
    selected = random.sample(range(len(synthetic_image_folders)), num)

    selected_img_folders = [synthetic_image_folders[i] for i in selected]
    selected_mask_folders = [synthetic_mask_folders[i] for i in selected]
    masks = []
    mask_names = []
    for k,mask_path in enumerate(selected_mask_folders):
        if os.path.isfile(mask_path+'/slices/0.png'):
            mask = io.imread(mask_path+'/slices/0.png')
            if len(mask.shape) > 2:
                mask = mask[...,0]
            mask = Image.fromarray(mask)
            mask = np.array(mask.resize((64,64),Image.NEAREST))
            mask[mask==255] = 0
            # look at the upper right corner to check if the mask is faulty
            sum = mask[-20:,:20] == patch
            if np.sum(sum) < 200:
                selected_img_folders.remove(image_paths+mask_path.split('/')[-1])
                continue
            masks.append(mask)
            mask_names.append(mask_path)
        else: selected_img_folders.remove(image_paths+mask_path.split('/')[-1])
    
    settings["numObjects"] = len(masks)
    settings["imageSize"] = (512-64, 708-64)
    settings["mask_names"] = mask_names
    positions, center_ellipsoid = CellPlacement(
        settings,
        cell_masks=masks,
        mu=10,
        sigma=0.2,
        cluster_prob=1.0,
    )
    settings["imageSize"] = (512, 708)
    kernel = np.ones((50, 50), np.uint8)
    center_ellipsoid = cv2.dilate(center_ellipsoid, kernel, iterations=1)
    # Create empty canvas
    img_canvas = np.ones(settings["imageSize"], dtype=np.float32)*23
    noise_locs = np.random.rand(2,2000)
    noise_locs[0,:] = noise_locs[0,:]*settings["imageSize"][0]//1
    noise_locs[1,:] = noise_locs[1,:]*settings["imageSize"][1]//1
    img_canvas[noise_locs.astype(int)[0,:], noise_locs.astype(int)[1,:]] = np.random.randint(low=20, high=500, size=(2000,))
    # add more to the center ellipsoid
    noise_locs = np.random.rand(2,15000)
    noise_locs[0,:] = noise_locs[0,:]*settings["imageSize"][0]//1
    noise_locs[1,:] = noise_locs[1,:]*settings["imageSize"][1]//1
    dummy_canvas = np.zeros(img_canvas.shape)
    a = np.zeros(img_canvas.shape)
    a[32:-32,32:-32] = center_ellipsoid
    center_ellipsoid = a
    dummy_canvas[noise_locs.astype(int)[0,:], noise_locs.astype(int)[1,:]] = np.random.randint(low=20, high=500, size=(15000,))
    dummy_canvas[center_ellipsoid==0] = 0
    img_canvas = img_canvas + dummy_canvas
    img_canvas = np.clip(img_canvas, a_min=0, a_max=None)
    mask_canvas = np.zeros(settings["imageSize"], dtype=np.float32)
    for i,position in enumerate(positions):
        if position[0] != -1:
            img = io.imread(selected_img_folders[i]+'/0.tif')
            img = Image.fromarray(img[0,...])
            indices = np.where(masks[i]!=0)
            img = np.array(img)
            # change the intensities randomly
            intensity = 1
            img_canvas[indices[0]+int(position[0]),indices[1]+int(position[1])] = np.clip(np.array(img[masks[i]!=0])*intensity, 0, 255)
            mask_canvas[indices[0]+int(position[0]),indices[1]+int(position[1])] = i + 1
    
    img_canvas = ndimage.convolve1d(img_canvas, weights=[1/20,2/20,3/20,4/20,4/20,3/20,2/20,1/20], axis=0)
    io.imsave(save_path + "01/" + str(j).zfill(4) + ".tif", img_canvas.astype(np.uint8))
    io.imsave(save_path + "01_ST/" + str(j).zfill(4) + ".tif", mask_canvas.astype(np.uint8))