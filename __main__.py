import sys
import os

import operator

import logging

import numpy as np

import warnings

import scipy
import scipy.io
import scipy.stats

from sklearn.exceptions import DataConversionWarning

from pprint import pprint

from models import *
from preprocessing import *
from utils import (ClassificationConfig,
                   load_dataset,
                   dataset_slicing,
                   digit_visualization,
                   dump_dataset,
                   preprocessing_chain_combinations)

logger = logging.getLogger(__name__)


def Model(name):
    """ Get imported model class by name string """
    return globals()[name]

if __name__ == "__main__":
    # Supress sklearn DataConversionWarning
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)


    # numpy print options
    np.set_printoptions(precision = 2, suppress = True, threshold=sys.maxsize, linewidth=200)

    # Log Level
    logging.basicConfig(level=logging.DEBUG)

    # Load config
    config = ClassificationConfig("config.ini")

    accuracies = []

    # Profiles
    for profile_name, profile in config.profiles.items():
        # Load dataset
        dataset = load_dataset(profile)

        digits_vec, digits_label, train_indices, test_indices = operator.itemgetter("digits_vec", "digits_labels", "trainset", "testset")(dataset)
        # Transpose X's, since numpy is row-based
        digits_vec = digits_vec.T
        digits_label = digits_label.T

        # trial
        training_set, testing_set = [dataset_slicing(digits_vec, digits_label, indices_set, index_start = 1) for indices_set in [train_indices, test_indices]]


        for sub_profile in preprocessing_chain_combinations(profile):

            # Experiment Trials
            for i, (train, test) in enumerate(zip(training_set, testing_set), 1):
                logger.info("Trial %d", i)

                # Create Model
                model = Model(name = profile["model_class"])(training_set = train, **sub_profile)

                results, testing_time = model.evaluate(*test)
                print ("profile: %s, model: %s, trial: %d, Accuracy: %.4f, training_time: %4.2f, testing_time: %4.2f, optional_chain: %s" % (profile_name, profile["model_class"], i, results, model.training_time, testing_time, model.preprocessor.optional_chain))

                accuracies.append(dict(profile=profile_name, model=profile["model_class"], trial=i, accuracy=results, training_time = model.training_time, testing_time = testing_time, preprocess_chain=model.preprocessor.optional_chain))


    print (json.dumps(accuracies))
                    
