import logging

import numpy as np

import scipy

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import Preprocessor


from utils import dump_dataset

logger = logging.getLogger(__name__)

class ClassificationModel:
    def __init__(self, preprocessing_chain, preprocessing_options, training_set, **kwargs):
        self.preprocessor = Preprocessor(preprocessing_chain, preprocessing_options, training_set)
        self.kwargs = kwargs
        self.__dict__.update(kwargs)
        if training_set is not None:
            self.fit(*training_set)

    def split(self, dataset):
        raise NotImplementedError
        return 

    def preprocess(self, X, Y, is_train = False):
        X = self.preprocessor(X)
        # dump_dataset(X, Y)
        return (X, Y.flatten() if Y is not None else Y)

    def _train(self, *argv, **kwargs):
        raise NotImplementedError

    def fit(self, *training_set):
        # Cross-validation
        training_set, validation_set = self.split(training_set)
        self.model = self._train(training_set, validation_set)

    def predict(self, test_X):
        """ Predicts the test """
        test_X, _ = self.preprocess(test_X, None)
        return self.model.predict(test_X)

    def evaluate(self, test_X, test_Y):
        prediction = self.predict(test_X)
        correct_count = np.count_nonzero(prediction == test_Y.flatten())
        print ("model: %s" % self.model_class)
        print (confusion_matrix(test_Y, prediction))
        print (classification_report(test_Y, prediction))
        return correct_count / float(len(test_Y))

class SVM(ClassificationModel):
    def fit(self, *training_set):
        train_X, train_Y = self.preprocess(*training_set, is_train = True)
        
        model = SVC(gamma = self.gamma, **self.best_parameters)

        if len(self.best_parameters) == 0:
            # Cross-validation is included
            model = GridSearchCV(model, self.param_grid, cv = self.cv, n_jobs = -1, refit = True, verbose = 3)
        # print (training_set)
        model.fit(train_X, train_Y)

        if len(self.best_parameters) == 0:
            # logger.info("GridSearchCV results: %s", dict(model.cv_results_))
            logger.info("GridSearchCV best_params_: %s", model.best_params_)
            logger.info("GridSearchCV best_estimator_: %s", model.best_estimator_)

        self.model = model


class KNN(ClassificationModel):
    def fit(self, *training_set):
        train_X, train_Y = self.preprocess(*training_set, is_train = True)
        
        # Default to metric = "minkowski" and p = 2 => Euclidean
        model = KNeighborsClassifier(n_neighbors = self.n_neighbors, n_jobs = -1)
        model.fit(train_X, train_Y)

        self.model = model

