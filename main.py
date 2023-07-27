import pandas as pd
import numpy as np
import random
import os
import shutil

import cv2
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'DejaVu Sans'
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

import tensorflow as tf

# import tensorflow as tf, re, math
# import tensorflow_addons as tfa
# import tensorflow.keras.backend as K
# import tensorflow_io as tfio
# import tensorflow_probability as tfp

# import yaml
# from IPython import display as ipd
# import json
# from datetime import datetime
#
# from glob import glob
# from tqdm.notebook import tqdm
# from kaggle_datasets import KaggleDatasets
# import sklearn
# from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# from sklearn.model_selection import KFold
# from sklearn.metrics import roc_auc_score
# from IPython import display as ipd
#
# import itertools
# import scipy
# import warnings
#
# # Show less log messages
# tf.get_logger().setLevel('ERROR')
# tf.autograph.set_verbosity(0)
#
# # Set true to show less logging messages
# os.environ["WANDB_SILENT"] = "true"
# import wandb