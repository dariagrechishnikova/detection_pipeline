# -*- coding: utf-8 -*-
"""losses.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qCXO1Do7SSW9vkzUNJuQt0OLvKqvIUmF
"""
import tensorflow as tf
from runner import *
from trainer import *
from data_provider import *
from custom_models import *
from custom_metrics import *
import os
import random
from tensorflow.keras.layers.experimental import preprocessing 
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers.merge import concatenate
from keras.utils.data_utils import get_file
from tensorflow import keras


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


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def Combo_loss(targets, inputs, eps=1e-9, ALPHA = 0.6, CE_RATIO = 0.5):
    #ALPHA < 0.5 penalises FP more, > 0.5 penalises FN more
    #CE_RATIO weighted contribution of modified CE loss compared to Dice loss
    smooth = 1
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)
    
    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, eps, 1.0 - eps)
    out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
    
    return combo



def iou_fn_loss(y_true, y_pred):
    n_classes = y_pred.shape[-1]
    intersection = tf.reduce_sum(y_true * y_pred, [1,2])
    neg_y_pred = 1 - y_pred
    false_negatives = tf.reduce_sum(y_true * y_pred, [1,2])
    all_true = tf.reduce_sum(y_true, [1,2])
    all_pred = tf.reduce_sum(y_pred, [1,2])
    iou = ((intersection + 1) / (all_true + all_pred + (10*false_negatives) - intersection + 1))
    weights = [0,1,1/4,1/10]
    iou_weighted = iou * weights
    return -tf.reduce_sum(iou_weighted, axis = [0,1])


def iou_loss(y_true, y_pred):
    n_classes = y_pred.shape[-1]
    intersection = tf.reduce_sum(y_true * y_pred, [1,2])
    all_true = tf.reduce_sum(y_true, [1,2])
    all_pred = tf.reduce_sum(y_pred, [1,2])
    iou = ((intersection + 1) / (all_true + all_pred - intersection + 1))
    return -tf.reduce_sum(iou, axis = [0,1])