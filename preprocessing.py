import numpy as np

import scipy
from scipy.ndimage import interpolation

def moment(image):
    x_len, y_len = image.shape
    # Create a mesh grid
    c = np.mgrid[:x_len, :y_len]
    # print (c)

    pixel_sum = image.sum()
    # Center of mass
    mu = np.array([np.sum(mesh * image) for mesh in c]) / pixel_sum
    mesh_zeromean = np.array([mesh - mu_component for (mesh, mu_component) in zip(c, mu)])
    var = np.array([np.sum(mesh ** 2 * image) for mesh in mesh_zeromean]) / pixel_sum

    # cov = np.sum((c[0] - mu[0]) * (c[1] - mu[1]) * image) / pixel_sum
    cov = np.sum(np.prod(mesh_zeromean, axis=0) * image) / pixel_sum

    # Covariance Matrix
    cov_matrix = np.array([[var[0], cov], [cov, var[1]]])

    # alpha
    alpha = cov / var[0]

    return mu, cov_matrix, alpha


def deskew(image, side_length = 28):
    r""" Apply affine transformation on / (Skewing back) the Image
    
    .. math:: 
        \text{Image}_\text{new} = A \text{Image}_0 + b

        \text{where} A = \

    """
    image = image.reshape((side_length, side_length)).T
    mu, cov_matrix, alpha = moment(image)
    affine_matrix = np.array([[1, 0], [alpha, 1]])
    offset = mu - affine_matrix @ (np.array(image.shape) / 2.0)

    return interpolation.affine_transform(image, affine_matrix, offset = offset)

def preprocess(image, **parameters):
    return deskew(image, **parameters)
