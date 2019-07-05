# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    height, width, channels = images[0].shape
    n = len(images)

    albedo = np.zeros((height, width, channels), dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            for k in range(channels):
                I = np.array([image[i, j, k] for image in images])
                I.reshape(-1, 1)
                print np.dot(lights.T,I).shape
                G = np.dot(np.linalg.inv(np.dot(lights, lights.T)), np.dot(lights.T, I))

                k_d = np.linalg.norm(G)

                if k_d < 1e-7:
                    albedo[i, j, k] = 0
                    normals[i, j] = 0

                else:
                    albedo[i, j, k] = k_d
                    normals[i, j] = G / k_d

    return albedo, normals


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    height, width = points.shape[0], points.shape[1]

    projections = np.zeros((height, width, 2))

    for j in range(width):
        for i in range(height):
            pts = np.append(points[i, j], 1)

            # Dot product with projection matrix(K*Rt)
            dot_project = np.dot(np.dot(K, Rt), pts)

            # Normalizing
            projections[i, j, 0] = dot_project[0] / dot_project[2]
            projections[i, j, 1] = dot_project[1] / dot_project[2]

    return projections



def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x112, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height, width, channels = image.shape
    normalized = np.zeros((height, width, channels * ncc_size * ncc_size), dtype=np.float32)
    half = ncc_size / 2

    for i in range(height):
        for j in range(width):
            if i - half < 0 or i + half >= height or j - half < 0 or j + half >= width:
                continue
            holder = list()
            for k in range(channels):
                selected = image[i - half: i + half + 1, j - half: j + half + 1, k]
                selected = (selected - np.mean(selected)).flatten()
                holder.append(selected.T)
            arr = np.hstack(tuple(holder)).flatten()
            norms = np.linalg.norm(arr)
            if norms < 1e-6:
                arr.fill(0)
            else:
                arr = arr / norms
            normalized[i, j] = arr
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height, width = image1.shape[0], image1.shape[1]

    ncc = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            ncc[i, j] = np.correlate(image1[i, j], image2[i, j])

    return ncc


