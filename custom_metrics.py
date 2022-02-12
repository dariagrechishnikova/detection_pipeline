# -*- coding: utf-8 -*-
"""metrics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1496W0ZmRU6S4eeAAm3gABtZW9YJ1w_me
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
from dataclasses import dataclass
from runner import *
from trainer import *
from initial_parser import *
from splitter import *
from data_provider import *
from custom_models import *
from custom_losses import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os
import random
from tensorflow.keras.layers.experimental import preprocessing 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from tensorflow.keras.applications import EfficientNetB0
from keras import backend as K
from keras.layers.merge import concatenate
from keras.utils.data_utils import get_file
from tensorflow import keras
import segmentation_models as sm
import cv2
import datetime
from segmentation_models.losses import bce_jaccard_loss

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def mean_iou(y_true, y_pred):
    n_classes = y_pred.shape[-1]
    intersection = tf.reduce_sum(y_true * y_pred, [1,2])
    all_true = tf.reduce_sum(y_true, [1,2])
    all_pred = tf.reduce_sum(y_pred, [1,2])
    iou = (intersection + 1) / (all_true + all_pred - intersection + 1)
    return tf.reduce_mean(iou, axis = [0,1])