import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Add, Conv2DTranspose
from keras.layers import Conv2D, SeparableConv1D
from keras.layers import Lambda
from keras.layers import Flatten, UpSampling1D
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2DTranspose
from keras import layers, models, optimizers
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers
from keras.models import load_model

from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, GlobalAveragePooling1D
import tensorflow.keras.backend as K
import numpy as np
from keras.layers import Conv2DTranspose, Bidirectional, GRU, LSTM, Input,Dense, SpatialDropout1D, Conv2D, MaxPooling2D, Flatten, Input, UpSampling2D, Dropout,Lambda, Average, concatenate, Activation, Add
import numpy as np
from keras.layers import Input,Dense, Add, UpSampling1D, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Input, UpSampling2D, Dropout,Lambda, Average, concatenate, Activation
from keras import optimizers, Model
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from sklearn.utils import class_weight
from numpy.random import seed
import math
from keras.regularizers import l2


from tensorflow.keras.layers import add, ConvLSTM2D, Reshape, Dense, AveragePooling2D, Input, Conv2DTranspose, TimeDistributed, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import add, Reshape, Dense, Input, TimeDistributed, Dropout, Activation, LSTM, Conv1D, Cropping1D
from tensorflow.keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
import locale
import os
import matplotlib

#matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from os import listdir, walk
from os.path import isfile, join, isdir
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
from datetime import datetime
from datetime import timedelta
import os.path
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
import csv

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, Flatten, Reshape, multiply
from keras.layers import concatenate, GRU, Input, LSTM, MaxPooling1D
from keras.layers import GlobalAveragePooling1D,  GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Model
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.models import Model

