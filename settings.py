import sys
sys.path.append('/content/drive/MyDrive/repos/image_processing_pipeline')


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
from runner import *
from trainer import *
from data_provider import *
from custom_models import *
from custom_metrics import *
from custom_losses import *
import os
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from tensorflow import keras
import cv2
import datetime
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import ensembling




log_dir = "tensor_board_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)




###setings for yolov3_detection sartorius


anchors = np.array([(9, 11), (11, 19), (16, 33), (18, 16), (26, 63),
                         (32, 27),(56, 43), (42, 114), (94, 78)],
                        np.float32) / 416 
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yloss = [YoloLoss(anchors[mask], classes=3) for mask in anchor_masks]

yolov3_sart_params = {
  'input_train_dataset': '/content/drive/MyDrive/Sartorius/tfrec/train.tfrec',
  'input_val_dataset': '/content/drive/MyDrive/Sartorius/tfrec/val.tfrec',
  'input_classes': '/content/drive/MyDrive/Sartorius/classes_names.txt', 
  'input_image_size': 416,
  'input_batch_size': 16,
  'input_yolo_max_boxes': 780,  
  'input_num_grid_cell': 32, 
  'input_buffer_size': 512, 
  'input_model_obj': yolov3(),
  'input_optimizer': tf.keras.optimizers.Adam(lr=0.0001), 
  'input_loss': yloss, 
  'input_metrics': 'acc', 
  'input_epochs': 100,  
  'net_info_path': '/content/drive/MyDrive/models_misc/network_info/', 
  'model_checkpoints_path': '/content/drive/MyDrive/models_misc/model_checkpoints',
  'input_callbacks': [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3), EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), tensorboard_callback],
}



###*****************************************************************************#####

params_dict = {'yolov3_sart' : yolov3_sart_params}

