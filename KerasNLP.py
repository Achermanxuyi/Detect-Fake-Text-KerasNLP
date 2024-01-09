import os
os.environ["KERAS_BACKED"] = "torch"

import keras_nlp
import keras_core as keras
import keras_core.backed as K

import jax
import tensorflow as tf

import numpy
import pandas as pd

import matplotlib.pyplot as plt

from glob import glob
from tqdm.notebook import tqdm
import gc

