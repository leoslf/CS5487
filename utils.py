import os
from os.path import dirname, realpath

import functools
from functools import partial, reduce

import configparser
import json

import itertools
import operator

import logging

import numpy as np

import scipy
import scipy.io

logger = logging.getLogger(__name__)

def vectorize(f):
    @functools.wraps(f)
    def wraps(self, X, *argv, **kwargs):
        return np.array(list(map(partial(f, self, *argv, **kwargs), X)))
        # return np.apply_along_axis(partial(f, self), np.arange(len(X.shape))[1:], X, *argv, **kwargs)
    return wraps

def log(f):
    @functools.wraps(f)
    def wraps(self, *argv, **kwargs):
        logger.debug(f.__name__)
        return f(self, *argv, **kwargs)
    return wraps

def compose(*functions):
    r""" Function Composition: :math:`(f_1 \circ \cdots \circ f_n)(x)` """
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

def methodgetter(obj, method_names):
    return map(partial(getattr, obj), method_names)

def working_dir(filename):
    return os.path.join(dirname(realpath(__file__)), filename)

def dataset_filenames(profile):
    return operator.itemgetter("dataset_filename", "challengeset_filename")(profile)

def describe_dataset(profile):
    filenames = dataset_filenames(profile)
    return list(map(scipy.io.whosmat, filenames))

def load_dataset(profile):
    filenames = dataset_filenames(profile)
    return list(map(scipy.io.loadmat, filenames))

def dataset_slicing(X, Y, indices_set, transpose=False, index_start = 0):
    if transpose:
        indices_set = indices_set.T
    indices_set = [np.array(indices, dtype=np.int) - index_start for indices in indices_set]
    return [(X[indices], Y[indices]) for indices in indices_set]

def digit_visualization(X, Y):
    print (X.reshape((28, 28)).T, Y)

def dump_dataset(X, Y):
    for (x, y) in zip(X, Y.flatten()):
        digit_visualization(x, y)

def preprocessing_chain_combinations(profile):
    a, b = profile["preprocessing_chain_after_squaring"], profile["preprocessing_chain_after_flattening"]
    merged = a + b
    merged_length = len(merged)
    
    # Handle case when preprocessing_combination_min == -1 => no subsets are tested
    preprocessing_combination_min = profile["preprocessing_combination_min"] 
    if preprocessing_combination_min < 0:
        preprocessing_combination_min = merged_length

    for r in range(preprocessing_combination_min, merged_length + 1):
        for indices in itertools.combinations(range(merged_length), r):
            a_indices, b_indices = [i for i in indices if len(indices) > 0 and i < len(a)], [i - len(a) for i in indices if len(indices) > 0 and i >= len(a)]

            subprofile = profile.copy()
            subprofile["preprocessing_chain_after_squaring"] = [a[i] for i in a_indices]
            subprofile["preprocessing_chain_after_flattening"] = [b[i] for i in b_indices]

            yield subprofile




def load_json(key, value):
    try:
        return json.loads(value)
    except json.decoder.JSONDecodeError as e:
        return value

class ClassificationConfig:
    def __init__(self, filename):
        self._filename = filename
        self.config = self.load_config()

    @property
    def filename(self):
        return working_dir(self._filename)

    def load_config(self):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(self.filename)
        return config

    def __getitem__(self, section_name):
        """ Return section """
        return { key: load_json(key, value) for (key, value) in self.config.items(section_name) }

    def __getattr__(self, section_name):
        """ Return section if exists """
        return self[section_name]

    @property
    def sections(self):
        return self.config.sections()

    @property
    def profiles(self):
        """ Return profiles """
        profile_names = list(filter(lambda s: s.startswith("profile"), self.sections))

        # NOTE: Excluded section "classification"
        # profiles = { name: self.classification.copy() for name in ["classification"] + profile_names }
        profiles = { name: self.classification.copy() for name in profile_names }

        for name, profile in profiles.items():
            # Override the default properties
            profile.update(self[name])
            if "dataset" in profile:
                profile["dataset_filename"] = self.common["filename"] % dict(name = profile["dataset"])
            if "challengeset" in profile:
                profile["challengeset_filename"] = self.common["filename"] % dict(name = profile["challengeset"])


        return { name: profile for (name, profile) in profiles.items() if profile["enabled"] }

