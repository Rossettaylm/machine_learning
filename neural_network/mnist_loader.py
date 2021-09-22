"""
mnist_loader
-------------


"""

import gzip

import _compat_pickle
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.  
    """
