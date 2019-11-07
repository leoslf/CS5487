import numpy as np

import scipy

class DigitSVM:
    def __init__(self, hyperparameters, training_set = None):
        self.__dict__.update(hyperparameters)
        if training_set is not None:
            self.fit(training_set)

    def split(self, dataset):
        raise NotImplementedError
        return 


    def fit(self, training_set):
        # Cross-validation
        training_set, validation_set = self.split(training_set)
        raise NotImplementedError

    def predict(self, test_X):
        """ Predicts the test """
        raise NotImplementedError

    def evaluate(self, testing_set):
        raise NotImplementedError


