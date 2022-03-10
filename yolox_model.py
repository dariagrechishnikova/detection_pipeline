from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)




yolo_max_boxes = 780
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5





def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = DarknetBlock(x, 256, 8)  # skip connection
    x =  DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, x, name=name)





def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv





def YoloOutput(filters,  classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x_cl = x_reg = DarknetConv(x, filters, 1)

        x_cl = DarknetConv(x_cl, filters, 3)
        x_cl = DarknetConv(x_cl, filters * 2, 3)
        x_cl = DarknetConv(x_cl, classes, 1, batch_norm=False)
        x_cl_output = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            classes)))(x_cl)


        x_reg = DarknetConv(x, filters, 3)
        x_reg = DarknetConv(x_reg, filters * 2, 3)
        x_reg = DarknetConv(x_reg, 5, 1, batch_norm=False)
        x_reg_output = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            5)))(x_reg)

        return tf.keras.Model(inputs, (x_cl_output, x_reg_output), name=name)(x_in)
    return yolo_output



def YoloV3(size=None, channels=3, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output = YoloOutput(512, classes, name='yolo_output')(x)

    

    return Model(inputs, output, name='yolox')


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


# As tensorflow lite doesn't support tf.size used in tf.meshgrid, 
# we reimplemented a simple meshgrid function that use basic tf function.
def _meshgrid(n_a, n_b):
  #Outputs a list of two  tensors with shapes [n_b, n_a]. Inside are grid indexes
  #Ex: _meshgrid(4, 3)
  #[<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
  #array([[0, 1, 2, 3],
  #      [0, 1, 2, 3],
  #     [0, 1, 2, 3]], dtype=int32)>,
  #<tf.Tensor: shape=(3, 4), dtype=int32, numpy=
  #array([[0, 0, 0, 0],
  #      [1, 1, 1, 1],
  #      [2, 2, 2, 2]], dtype=int32)>]

    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]



def YoloXLoss(classes, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class = tf.split(
            y_pred, (4, 1, classes), axis=-1)
        pred_xy = pred_box[..., 0:2]
        pred_wh = pred_box[..., 2:4]
        pred_x1y1 = pred_xy - pred_wh / 2
        pred_x2y2 = pred_xy + pred_wh / 2
        pred_box = tf.concat([pred_x1y1, pred_x2y2], axis=-1)

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = true_box[..., 0:2]
        true_wh = true_box[..., 2:4]
        true_x1y1 = true_xy - true_wh / 2
        true_x2y2 = true_xy + true_wh / 2
        true_box = tf.concat([true_x1y1, true_x2y2], axis=-1)

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                #take elements from tensor on positions where mask 
                #elements are True. Mask dim <= tensor dim
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 4. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss



def myyolox(classes = 3):
  x = inputs = Input([None, None, 3])

  x = tf.keras.layers.Conv2D(filters = 2, kernel_size = 3, strides=(4, 4), activation='swish')(x)
  x = tf.keras.layers.Conv2D(filters = (classes + 5), kernel_size = 1, strides=(4, 4), activation='swish')(x)
  x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], classes + 5)))(x)

  return tf.keras.Model(inputs, x)



def myConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding, activation='swish',
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
    return x


def myResidual(x, filters):
    prev = x
    x = myConv(x, filters // 2, 1)
    x = myConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def myBlock(x, filters, blocks):
    x = myConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = myResidual(x, filters)
    return x


def myconv_res(classes=3):
    x = inputs = Input([None, None, 3])
    x = myConv(x, 32, 3)
    x = myBlock(x, 64, 1)
    x = myBlock(x, 128, 1)  # skip connection
    x = myBlock(x, 256, 1)  # skip connection
    x =  myBlock(x, 512, 1)

    x = tf.keras.layers.Conv2D(filters = (classes + 5), kernel_size = 3, strides=1, padding='same', activation='swish')(x)
    x = tf.keras.layers.Conv2D(filters = (classes + 5), kernel_size = 1, strides=1, padding='same', activation='swish')(x)
    x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], classes + 5)))(x)
    
    return tf.keras.Model(inputs, x)


  
