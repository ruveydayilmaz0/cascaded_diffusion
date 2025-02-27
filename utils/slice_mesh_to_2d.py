import open3d as o3d
import numpy as np
import cv2
from skimage import io
import glob
import os
import argparse



def parse_args():
       parser = argparse.ArgumentParser()
       parser.add_argument('--root_volumes', type=str, required=True, help='Path to the root folder containing the volumes')
       parser.add_argument('--output_size', type=int, default=256, help='Desired output size of the slices')
       return parser.parse_args()

def slice_volumes(args):
       folders = sorted(glob.glob(args.root_volumes+'*'))
       for folder in folders:
              pcd = o3d.io.read_point_cloud(folder+"/neus/mesh.ply")
              # Extract points from the Open3D point cloud
              points = np.asarray(pcd.points)
              # get certain points x and y
              points = points - np.min(points, axis=0)
              max_z = np.max(points[:,2])
              z_levels = np.linspace(max_z/2, max_z, 10)
              # create a folder to save the created slices
              slices_path = folder+"/slices/"
              if not os.path.exists(slices_path): 
                     os.makedirs(slices_path)
              
              for i,z_level in enumerate(z_levels):
                     img = np.zeros((args.output_size, args.output_size), dtype=np.uint8)
                     borders = (points[np.abs(points[:,2]-z_level)<0.1]*200)[:,:2].astype(np.uint8)
                     img[borders[:,0]+20, borders[:,1]+20] = 255
                     # Fill in the shape
                     contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                     im = np.zeros(img.shape + (3,))
                     cv2.drawContours(im, contours, -1, (128,128,128), thickness=cv2.FILLED)
                     io.imsave(slices_path+str(i)+'.png', im.astype(np.uint8))


if __name__ == "__main__":
    args = parse_args()
    slice_volumes(args)