import sys
import os

import operator

import logging

import numpy as np

import scipy
import scipy.io
import scipy.stats

from pprint import pprint

from model import *
from utils import (ClassificationConfig,
                   load_dataset,
                   dataset_slicing,
                   digit_visualization)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Log Level
    logging.basicConfig(level=logging.INFO)

    # Load config
    config = ClassificationConfig("config.ini")

    for profile_name, profile in config.profiles.items():
        # Load dataset
        dataset = load_dataset(profile)

        digits_vec, digits_label, train_indices, test_indices = operator.itemgetter("digits_vec", "digits_labels", "trainset", "testset")(dataset)
        digits_vec = digits_vec.T
        digits_label = digits_label.T

        print (digits_vec.shape)

        # trial
        training_set, testing_set = [dataset_slicing(digits_vec, digits_label, indices_set, transpose=True, index_start = 1) for indices_set in [train_indices, test_indices]]

        np.set_printoptions(threshold=sys.maxsize, linewidth=200)

        # pprint (training_set)
        # pprint (testing_set)
        train_X, train_Y = list(zip(*training_set))
        for iteration in zip(train_X, train_Y):
            for (X, Y) in zip(*iteration):
                digit_visualization(X, Y)


        for train, test in zip(training_set, testing_set):
            # Experiment Trial
            model = DigitSVM(train)

            results = model.evaluate(test)
            print (results)
            
