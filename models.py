import logging

import time

import numpy as np

import scipy

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import Preprocessor


from utils import dump_dataset

logger = logging.getLogger(__name__)

class ClassificationModel:
    def __init__(self, preprocessing_chain_after_squaring, preprocessing_chain_after_flattening, preprocessing_options, training_set, **kwargs):
        self.preprocessor = Preprocessor(preprocessing_chain_after_squaring, preprocessing_chain_after_flattening, preprocessing_options, training_set)
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

    # def fit(self, *training_set):
    #     # Cross-validation
    #     training_set, validation_set = self.split(training_set)
    #     self.model = self._train(training_set, validation_set)

    def predict(self, test_X):
        """ Predicts the test """
        #test_X, _ = self.preprocess(test_X, None)
        return self.model.predict(test_X)

    def evaluate(self, test_X, test_Y):
        
        test_X, _ = self.preprocess(test_X, None)
        
        start_time = time.time()
        prediction = self.predict(test_X)
        end_time = time.time()

        correct_count = np.count_nonzero(prediction == test_Y.flatten())
        print ("model: %s" % self.model_class)
        print (confusion_matrix(test_Y, prediction))
        print (classification_report(test_Y, prediction))
        return correct_count / float(len(test_Y)), end_time - start_time

    @property
    def grid_search(self) -> bool:
        return hasattr(self, "param_grid") and len(self.best_parameters) == 0

    @property
    def base_model(self):
        raise NotImplementedError

    def fit(self, *training_set):
        
        preprocessing_start = time.time()
        train_X, train_Y = self.preprocess(*training_set, is_train = True)
        preprocessing_end = time.time()
        self.preprocessing_time = preprocessing_end - preprocessing_start
        
        model = self.base_model

        if self.grid_search:
            # Cross-validation is included
            model = GridSearchCV(model, self.param_grid, **self.gridsearch_param)

        start_time = time.time()
        model.fit(train_X, train_Y)
        end_time  = time.time()
        self.training_time = end_time - start_time

        if self.grid_search:
            logger.info("GridSearchCV best_params_: %s", model.best_params_)
            logger.info("GridSearchCV best_estimator_: %s", model.best_estimator_)

            self.training_time = model.cv_results_["mean_fit_time"][model.best_index_]

        #self.training_time += preprocessing_end - preprocessing_start
        
        self.model = model


class SVM(ClassificationModel):
    @property
    def base_model(self):
        return SVC(gamma = self.gamma, **self.best_parameters)


class KNN(ClassificationModel):
    @property
    def base_model(self):
        # Default to metric = "minkowski" and p = 2 => Euclidean
        return KNeighborsClassifier(n_jobs = -1, **self.best_parameters)

class FishersDiscriminant(ClassificationModel):
    @property
    def base_model(self):
        return LinearDiscriminantAnalysis(**self.best_parameters)

class NaiveBayesGaussian(ClassificationModel):
    @property
    def base_model(self):
        return GaussianNB(**self.best_parameters)
