import os
from os.path import dirname, realpath

import configparser
import json

import logging

import numpy as np

import scipy
import scipy.io

logger = logging.getLogger(__name__)

def working_dir(filename):
    return os.path.join(dirname(realpath(__file__)), filename)

def load_dataset(profile):
    return scipy.io.loadmat(profile["dataset_filename"])

def dataset_slicing(X, Y, indices_set, transpose=False, index_start = 0):
    if transpose:
        indices_set = indices_set.T
    indices_set = [np.array(indices, dtype=np.int) - index_start for indices in indices_set]
    return [(X[indices], Y[indices]) for indices in indices_set]

def digit_visualization(X, Y):
    print (X.reshape((28, 28)).T, Y)

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

        profiles = { name: self.classification.copy() for name in ["classification"] + profile_names }

        for name, profile in profiles.items():
            # Override the default properties
            profile.update(self[name])
            if "dataset" in profile:
                profile["dataset_filename"] = self.common["filename"] % dict(name = profile["dataset"])

        return profiles

