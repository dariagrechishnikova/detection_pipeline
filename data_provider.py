# -*- coding: utf-8 -*-
"""data_provider.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15zegQhVM2kYWQ1G61M-O3Zj_qm9x5EfJ
"""

import tensorflow as tf
from custom_models import *
from custom_metrics import *
from custom_losses import *
import os
import random 
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
from keras import backend as K
import cv2


class target_transforms():
  @tf.function
  def transform_targets_for_output(self, y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


  def transform_targets(self, y_train, anchors, anchor_masks, size, num_grid_cell):
    y_outs = []
    grid_size = size // num_grid_cell

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(self.transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


  def transform_images(self, x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


  



class parser():
  def __init__(self):
    self.IMAGE_FEATURE_MAP = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }
  def parse_tfrecord(self, tfrecord, class_table, size, yolo_max_boxes):
    x = tf.io.parse_single_example(tfrecord, self.IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


  def load_tfrecord_dataset(self, file_pattern, class_file, yolo_max_boxes, size):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: self.parse_tfrecord(x, class_table, size, yolo_max_boxes))



class data_povider_detection():
  def __init__(self, input_train_dataset, input_val_dataset, input_classes, 
  input_image_size, input_batch_size, input_yolo_max_boxes, input_num_grid_cell,
  input_buffer_size = 512):
    self.tt_obj = target_transforms()
    self.parser_obj = parser()
    self.train_dataset = input_train_dataset
    self.val_dataset = input_val_dataset
    self.classes = input_classes
    self.size = input_image_size
    self.buffer_size = input_buffer_size
    self.batch_size = input_batch_size
    self.yolo_max_boxes = input_yolo_max_boxes
    self.num_grid_cell = input_num_grid_cell
    self.anchors = np.array([(9, 11), (11, 19), (16, 33), (18, 16), (26, 63),
                         (32, 27),(56, 43), (42, 114), (94, 78)],
                        np.float32) / self.size              
    self.anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

  def get_train(self):
    train_dataset = self.parser_obj.load_tfrecord_dataset(self.train_dataset, self.classes, self.yolo_max_boxes, self.size)
    train_dataset = train_dataset.shuffle(self.buffer_size)
    train_dataset = train_dataset.batch(self.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        self.tt_obj.transform_images(x, self.size),
        self.tt_obj.transform_targets(y, self.anchors, self.anchor_masks, self.size, self.num_grid_cell)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset

  def get_val(self):
    val_dataset = self.parser_obj.load_tfrecord_dataset(self.val_dataset, self.classes, self.yolo_max_boxes, self.size)
    val_dataset = val_dataset.batch(self.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        self.tt_obj.transform_images(x, self.size),
        self.tt_obj.transform_targets(y, self.anchors, self.anchor_masks, self.size, self.num_grid_cell)))
    return val_dataset







  