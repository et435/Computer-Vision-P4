#ET435 CG595
import numpy as np
import cv2


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

    height = images[0].shape[0]
    width = images[0].shape[1]
    channels = images[0].shape[2]

    albedo = np.zeros((height,width,channels), dtype = "float32")
    normal = np.zeros((height,width,3), dtype = "float32")

    for x in range(height):
        for y in range(width):
            for z in range(channels):

                image = (np.array([individual_image[x,y,z] for individual_image in images]))
                image.reshape(-1,1)

                left = np.dot(lights.T,lights)
                left_inv = np.linalg.inv(left)
                right = np.dot(lights.T,image)
                G = np.dot(left_inv,right)
                k = np.linalg.norm(G)

                if k < 1e-7:
                    albedo[x,y,z] = 0
                    normal[x,y] = 0
                else:
                    albedo[x,y,z] = k
                    normal[x,y] = (G/k)

    return albedo, normal




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

    height = points.shape[0]
    width = points.shape[1]
    projection_norm = np.zeros((height,width,2))

    for x in range(height):
        for y in range(width):
            points_indv = np.append(points[x,y],1)
            KRt = np.dot(K,Rt)
            projection = np.dot(KRt,points_indv)
            for img in range(2):
                projection_norm[x,y,img] = (projection[img]/projection[2])

    return projection_norm



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
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    normalized_array = np.zeros((height, width, channels * ncc_size**2), dtype = "float32")
    ncc_half = (ncc_size/2)

    for x in range(height):
        for y in range(width):

            array_list = list()

            if (x < ncc_half) or (y < ncc_half):
                continue
            if (x >= height-ncc_half) or (y >= width-ncc_half):
                continue

            for z in range(channels):

                interest = image[x-ncc_half:x+ncc_half+1, y-ncc_half:y+ncc_half+1, z]
                subtracted = interest-np.mean(interest)
                flattened = subtracted.flatten()
                transposed = flattened.T
                array_list.append(transposed)

            array_stacked = np.hstack(tuple(array_list))
            array_flattened = array_stacked.flatten()
            normalized_val = np.linalg.norm(array_flattened)

            if normalized_val >= 1e-6:
                array_flattened = array_flattened / normalized_val
            else:
                array_flattened.fill(0)

            normalized_array[x,y] = array_flattened

    return normalized_array


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

    height = image1.shape[0]
    width = image1.shape[1]
    ncc = np.zeros((height,width))

    for x in range(height):
        for y in range(width):
            correlate_vals = np.correlate(image1[x,y],image2[x,y])
            ncc[x,y] = correlate_vals[0]
    return ncc
