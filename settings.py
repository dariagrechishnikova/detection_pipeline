import sys
sys.path.append('/content/drive/MyDrive/repos/image_processing_pipeline')


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
from custom_metrics import *
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
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ensembling




log_dir = "tensor_board_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


###setings for ZFTurbo Unet
ZFTurbo_Unet_bce_loss_adam_cl1_params = {
  'initial_split': 'Y',
  'class_num': 1,
  'batch_size': 32,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3), EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 500, 
  'img_size': 224, 
  'label_path': '/content/label/', 
  'loss': tf.keras.losses.BinaryCrossentropy(), 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': ZF_UNET_224(input_channels = 3, output_mask_channels = 1), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}


###setings for Ternaus Unet
Ternaus_bce_loss_adam_cl1_params = {
  'initial_split': 'N',
  'class_num': 1,
  'batch_size': 16,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3), EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 500, 
  'img_size': 256, 
  'label_path': '/content/label/', 
  'loss': tf.keras.losses.BinaryCrossentropy(), 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': ternaus_net(), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}


simple_unet_bce_loss_adam_cl1_params = {
  'initial_split': 'N',
  'class_num': 1,
  'batch_size': 11,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 500, 
  'img_size': 224, 
  'label_path': '/content/label/', 
  'loss': tf.keras.losses.BinaryCrossentropy(), 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': simple_unet(), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}


vgg16_unet_bce_jaccard_loss_adam_cl1_params = {
  'initial_split': 'N',
  'class_num': 1,
  'batch_size': 32,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 500, 
  'img_size': 224, 
  'label_path': '/content/label/', 
  'loss': bce_jaccard_loss, 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': vgg16_bb_unet(output_mask_channels=1), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}




seresnext50_unet_bce_jaccard_loss_adam_cl1_params = {
  'initial_split': 'N',
  'class_num': 1,
  'batch_size': 32,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 500, 
  'img_size': 224, 
  'label_path': '/content/label/', 
  'loss': bce_jaccard_loss, 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': seresnext50_bb_unet(output_mask_channels = 1), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}


densenet_fpn_bce_jaccard_loss_adam_cl1_params = {
  'initial_split': 'N',
  'class_num': 1,
  'batch_size': 11,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 500, 
  'img_size': 224, 
  'label_path': '/content/label/', 
  'loss': bce_jaccard_loss, 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': fpn_bb_unet(output_mask_channels = 1), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}


custom_fpn_bce_jaccard_loss_adam_cl1_params = {
  'initial_split': 'N',
  'class_num': 1,
  'batch_size': 16,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 500, 
  'img_size': 224, 
  'label_path': '/content/label/', 
  'loss': bce_jaccard_loss, 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': custom_fpn(), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}


custom_cnn_bce_jaccard_loss_adam_cl1_params = {
  'initial_split': 'N',
  'class_num': 1,
  'batch_size': 16,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 500, 
  'img_size': 224, 
  'label_path': '/content/label/', 
  'loss': bce_jaccard_loss, 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': custom_cnn(), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}




###setings for ZFTurbo Unet
trained_model1 = tf.keras.models.load_model('/content/drive/MyDrive/models_misc/model_checkpoints/model_ZFTurbo_Unet_sart_fold_0.hdf5',
                                    custom_objects={'jacard_coef_loss': jacard_coef_loss, 
                                                    'jacard_coef': jacard_coef})

ZFTurbo_Unet_sart_params = {
  'initial_split': 'Y',
  'class_num': 1,
  'height': 224, 
  'width': 224,
  'batch_size': 32,
  'buffer_size': 1000,
  'callbacks': [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3), EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), tensorboard_callback],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/boxed_images/cort/', 
  'epochs': 500, 
  'img_size': 224, 
  'label_path': '/content/boxed_masks/cort/', 
  'loss': jacard_coef_loss, 
  'metrics': [jacard_coef], 
  'model_obj': ZF_UNET_224(input_channels = 3, output_mask_channels = 1), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/MyDrive/models_misc/tr_val_split_binary_from_multiclass_cl1', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test_cl1',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model_cl1',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': trained_model1,
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation_cl1.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': A.CLAHE(p=1)

}



####settings for ensemble
###code example
'''trained_model1 = tf.keras.models.load_model('/content/drive/MyDrive/models_misc/model_checkpoints/model_densenet_fpn_bce_jaccard_loss_adam_cl1_fold_0.hdf5',
                                    custom_objects={'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss, 
                                                    'jacard_coef': jacard_coef, 'mean_iou': mean_iou, 'dice_coef': dice_coef})
trained_model2 = tf.keras.models.load_model('/content/drive/MyDrive/models_misc/model_checkpoints/model_seresnext50_unet_bce_jaccard_loss_adam_cl1_fold_0.hdf5',
                                    custom_objects={'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
                                                    'jacard_coef': jacard_coef, 'mean_iou': mean_iou, 'dice_coef': dice_coef})
trained_model3 = tf.keras.models.load_model('/content/drive/MyDrive/models_misc/model_checkpoints/model_vgg16_unet_bce_jaccard_loss_adam_cl1_fold_0.hdf5',
                                    custom_objects={'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss, 
                                                    'jacard_coef': jacard_coef, 'mean_iou': mean_iou, 'dice_coef': dice_coef})
trained_model4 = tf.keras.models.load_model('/content/drive/MyDrive/model_checkpoints_fetreg/weights_fold_15.hdf5',
                                    custom_objects={'Combo_loss': Combo_loss, 
                                                    'jacard_coef': jacard_coef, 'mean_iou': mean_iou, 'dice_coef': dice_coef})




ensemble_params = {
  'initial_split': 'N',
  'class_num': 1,
  'batch_size': 32,
  'buffer_size': 1000,
  'callbacks': [],  
  'class_count': 4, 
  'cv_folds': 5, 
  'data_path': '/content/drive/MyDrive/dedups/', 
  'epochs': 32, 
  'img_size': 224, 
  'label_path': '/content/label/', 
  'loss': iou_fn_loss, 
  'metrics': [jacard_coef, mean_iou], 
  'model_obj': efficientnet_bb_unet(output_mask_channels = 4), 
  'optimizer': 'adam', 
  'seed': 0, 
  'test_size': 0.2, 
  'val_size': 0.1, 
  'tr_val_split_file': '/content/drive/models_misc/tr_val_split', 
  'test_file': '/content/drive/MyDrive/models_misc/test_split_binary_from_multiclass_cl1', 
  'net_info_path': '/content/drive/MyDrive/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/model_checkpoints',
  'second_layer_test_size': 0.4,
  'final_test_path': '/content/drive/MyDrive/models_misc/final_test',
  'second_layer_model_path': '/content/drive/MyDrive/models_misc/second_layer_model',
  'second_layer_model_obj': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
  'nn_zoo': [trained_model1, trained_model2, trained_model3, trained_model4],
  'item_range': 6,
  'eval_file': '/content/drive/MyDrive/models_misc/evaluation.txt',  
  'ttas_list': [A.HorizontalFlip(p=1),A.VerticalFlip(p=1),A.Transpose(p=1)],
  'permanent_tta': None

}
'''



###*****************************************************************************#####

params_dict = {'ZFTurbo_Unet_sart' : ZFTurbo_Unet_sart_params}

