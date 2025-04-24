import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
import cv2

"""
This function was implemented using the work Svoboda, D., Ulman, V. (2013). Towards a Realistic Distribution of Cells in Synthetically Generated 3D Cell Populations. In: Petrosino, A. (eds) Image Analysis and Processing – ICIAP 2013. ICIAP 2013. Lecture Notes in Computer Science, vol 8157. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-41184-7_44.
Cells are iteratively placed according to clusteing probability. Cross correlation is used first to eliminate cells intersecting on placement. If a cell is going to be attached to a cluster distance mapping is used so it is close to one of the already assigned cells.
"""

def create_center_ellipsoid(image):
    # Center of the ellipse
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2

    # Major and minor radii of the ellipse
    radius_y, radius_x = 100, 150

    # Loop over every pixel and check if it lies within the ellipse
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Equation of an ellipse ((x-h)^2 / a^2) + ((y-k)^2 / b^2) <= 1
            if ((x - center_x) ** 2) / radius_x ** 2 + ((y - center_y) ** 2) / radius_y ** 2 <= 1:
                image[y, x] = 1
    return image

def CellPlacement(settings, cell_masks, mu, sigma, cluster_prob, save_results=False):
    FOV = np.zeros(settings["imageSize"], dtype=np.int16)
    cell_placements = np.zeros((settings["numObjects"], 2))
    k = 0
    for mask in cell_masks:
        # print(settings['mask_names'][k])
        mask[mask > 0] = 1
        # mask = gaussian_filter(mask, sigma=1) # might be necessary for some datasets
        region_props = regionprops(mask.astype(int))
        rangeY = np.arange(
            np.round(region_props[0].bbox[1]), np.round(region_props[0].bbox[3])
        )
        rangeX = np.arange(
            np.round(region_props[0].bbox[0]), np.round(region_props[0].bbox[2])
        )
        diagonal_length = int(
            np.round((np.sqrt((len(rangeX) ** 2 + len(rangeY) ** 2))))
        )

        if k == 0:
            placement = [np.random.choice(np.arange(200,settings["imageSize"][0]-200), size=1), \
                         np.random.choice(np.arange(300,settings["imageSize"][1]-300), size=1)]
            region_props = regionprops(mask.astype(int))
            rangeY = np.arange(
                np.round(region_props[0].bbox[1]), np.round(region_props[0].bbox[3])
            )
            rangeX = np.arange(
                np.round(region_props[0].bbox[0]), np.round(region_props[0].bbox[2])
            )
            rangeXGlobal = np.arange(
                placement[0] - len(rangeX) / 2,
                placement[0] + len(rangeX) / 2,
                dtype=np.int16,
            )
            rangeYGlobal = np.arange(
                placement[1] - len(rangeY) / 2,
                placement[1] + len(rangeY) / 2,
                dtype=np.int16,
            )
            # clip the global range with image size
            rangeXGlobal = rangeXGlobal[rangeXGlobal<settings["imageSize"][0]]
            rangeYGlobal = rangeYGlobal[rangeYGlobal<settings["imageSize"][1]]
            for x_global, x in zip(rangeXGlobal, rangeX):
                for y_global, y in zip(rangeYGlobal, rangeY):
                    FOV[x_global, y_global] = np.maximum(
                        FOV[x_global, y_global], mask[x, y]
                    )
            cell_placements[k, :] = placement
            k += 1
        else:
            padded_mask = np.zeros(
                (settings["imageSize"][0], settings["imageSize"][1]), dtype=np.int16
            )
            padded_mask[
                int((padded_mask.shape[0] / 2) - (mask.shape[0] / 2)) : int(
                    (padded_mask.shape[0] / 2) + (mask.shape[0] / 2)
                ),
                int((padded_mask.shape[1] / 2) - (mask.shape[1] / 2)) : int(
                    (padded_mask.shape[1] / 2) + (mask.shape[1] / 2)
                ),
            ] = mask
            F1 = fft2(padded_mask)
            F2 = fft2(FOV)
            cross_power_spectrum = F1 * F2
            cross_correlation = ifft2(cross_power_spectrum)
            cross_correlation = fftshift(cross_correlation)
            cross_correlation = np.abs(cross_correlation)
            cross_correlation /= np.max(cross_correlation)
            cross_correlation[cross_correlation < sigma] = 10
            cross_correlation[cross_correlation < 10] = 0
            cross_correlation[cross_correlation == 10] = 1
            locations = cross_correlation
            locations[locations > 0] = 1

            # decide if new cell joins a cluster
            p = np.random.uniform()
            if p <= cluster_prob:
                distance_map = distance_transform_edt(1 - locations)
                inverted_distance_map = 255 - distance_map
                ke = np.random.normal(loc=20 + diagonal_length, scale=10)
                if ke <= 10 + diagonal_length:
                    ke = 50 + diagonal_length//2
                elif ke >= 70 + diagonal_length:
                    ke = 20 + diagonal_length
                kernel = np.ones((50, 50), np.uint8)
                eroded_image = cv2.erode(inverted_distance_map, kernel, iterations=1)
                distance_mask = np.logical_and(
                    eroded_image < 254, eroded_image > 255 - mu
                )
                eroded_image[distance_mask] = -100
                eroded_image[eroded_image > 0] = 0
                eroded_image[eroded_image == -100] = 1
                candidates = eroded_image.astype(int)

                # we dont want cells too near the edges
                center_rectangle = np.zeros(
                    (settings["imageSize"][0], settings["imageSize"][1]), dtype=np.int16
                )
                center_rectangle = create_center_ellipsoid(center_rectangle)
                candidates = np.bitwise_and(
                    center_rectangle.astype(np.uint8), candidates.astype(np.uint8)
                )
            else:
                candidates = locations
                ke = np.random.normal(loc= diagonal_length, scale=10)
                if ke <= 10 + diagonal_length:
                    ke = diagonal_length//2
                elif ke >= 70 + diagonal_length:
                    ke = diagonal_length
                kernel = np.ones((2,2), np.uint8)
                # kernel = np.ones((int(ke), int(ke)), np.uint8)
                candidates = cv2.erode(candidates, kernel, iterations=1)
                # we dont want cells too near the edges
                center_rectangle = np.zeros(
                    (settings["imageSize"][0], settings["imageSize"][1]), dtype=np.int16
                )
                center_rectangle = create_center_ellipsoid(center_rectangle)
                candidates = np.bitwise_and(
                    center_rectangle.astype(np.uint8), candidates.astype(np.uint8)
                )

            coords = np.column_stack(np.where(candidates == 1))
            # Choose a random pixel using np.random.choice
            if len(coords) != 0:
                random_index = np.random.choice(len(coords))
                placement = coords[random_index]

                rangeXGlobal = np.arange(
                    placement[0] - len(rangeX) / 2,
                    placement[0] + len(rangeX) / 2,
                    dtype=int,
                )
                rangeYGlobal = np.arange(
                    placement[1] - len(rangeY) / 2,
                    placement[1] + len(rangeY) / 2,
                    dtype=int,
                )
                # clip the global range with image size
                rangeXGlobal = rangeXGlobal[rangeXGlobal<settings["imageSize"][0]]
                rangeYGlobal = rangeYGlobal[rangeYGlobal<settings["imageSize"][1]]
                
                for x_global, x in zip(rangeXGlobal, rangeX):
                    for y_global, y in zip(rangeYGlobal, rangeY):
                        FOV[x_global, y_global] = np.maximum(
                            FOV[x_global, y_global], mask[x, y]
                        )
                cell_placements[k, :] = placement
            else:
                cell_placements[k, :] = np.array([-1, -1])
            
            if save_results:
                cell_placements[k, :] = placement
                result_image = np.zeros((settings["imageSize"][0], settings["imageSize"][1], 3))
                result_image[FOV > 0, :] = 255
                result_image[candidates > 0, 1] = 255
                result_image[placement, 0] = 255

            k += 1
    return cell_placements, center_rectangle