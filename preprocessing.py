import numpy as np

import scipy
from scipy.ndimage import interpolation, gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import (compose, methodgetter, vectorize, log)


class Preprocessor:
    """ Encapsulation of Preprocessing Procedures """
    def __init__(self, methodname_chain, options, training_set):
        self.methodname_chain = methodname_chain
        self.__dict__.update(options)
        train_X, train_Y = self.training_set = training_set
        self.scaler = StandardScaler().fit(train_X)
        self.PCA = PCA().fit(train_X)

    def _moment(self, image):
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


    # @log
    @vectorize
    def deskewing(self, image, side_length = 28):
        r""" Apply affine transformation on / (Skewing back) the Image
        
        .. math:: 
            \text{Image}_\text{new} = A \text{Image}_0 + b

            \text{where} A =    \begin{bmatrix}
                                    1 & 0 \\
                                    \alpha & 1
                                \end{bmatrix}, and \\
                        \alpha = \frac{Cov(X, Y)}{Var(X)}

        """
        # print (image)
        # image = image.reshape((side_length, side_length)).T
        mu, cov_matrix, alpha = self._moment(image)
        affine_matrix = np.array([[1, 0], [alpha, 1]])
        offset = mu - affine_matrix @ (np.array(image.shape) / 2.0)

        return interpolation.affine_transform(image, affine_matrix, offset = offset)

    # @log
    @vectorize
    def squaring(self, x):
        # print (x.shape)
        return x.reshape(self.img_shape, order="F")

    # @log
    @vectorize
    def flattening(self, x):
        return x.flatten(order="F")

    def blurring(self, x):
        return gaussian_filter(x, self.blur_sigma)

    # @log
    def normalizing(self, X):
        return self.scaler.transform(X)

    def pca(self, X):
        return self.PCA.transform(X)

    @property
    def method_chain(self):
        """ Return a function composed by the methods specified by methodname_chain """
        return compose(*methodgetter(self, self.methodname_chain))

    def __call__(self, X):
        """ Executing the function composition chain in order to preprocess the dataset """
        return self.method_chain(X)
