import sys
import os

import operator

import logging

import numpy as np

import scipy
import scipy.io
import scipy.stats

from pprint import pprint

from models import *
from preprocessing import *
from utils import (ClassificationConfig,
                   load_dataset,
                   dataset_slicing,
                   digit_visualization,
                   dump_dataset)

logger = logging.getLogger(__name__)


def Model(name):
    return globals()[name]

if __name__ == "__main__":
    # Log Level
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)

    # Load config
    config = ClassificationConfig("config.ini")

    for profile_name, profile in config.profiles.items():
        # Load dataset
        dataset = load_dataset(profile)

        digits_vec, digits_label, train_indices, test_indices = operator.itemgetter("digits_vec", "digits_labels", "trainset", "testset")(dataset)
        digits_vec = digits_vec.T
        digits_label = digits_label.T

        # print (digits_vec.shape)

        # trial
        training_set, testing_set = [dataset_slicing(digits_vec, digits_label, indices_set, index_start = 1) for indices_set in [train_indices, test_indices]]
        # print (training_set, testing_set)

        np.set_printoptions(precision = 2, suppress = True, threshold=sys.maxsize, linewidth=200)

        # Experiment Trials
        for i, (train, test) in enumerate(zip(training_set, testing_set), 1):
            logger.info("Trial %d", i)

            # Create Model
            model = Model(name = profile["model_class"])(training_set = train, **profile)

            results = model.evaluate(*test)
            print ("model: %s, trial: %d, Accuracy: %4.2f" % (profile["model_class"], i, results))
                
