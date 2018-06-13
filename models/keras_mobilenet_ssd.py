'''
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari, Aiken Cairncross

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, BatchNormalization, Conv2D, Reshape, Concatenate, DepthwiseConv2D
from keras.regularizers import l2
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

# You need to import keras before importing keras_applications
import keras
# It's probably better to use keras.applications somehow, but the mobilenet
# modules in there don't include relu6
from keras_applications import mobilenet, mobilenet_v2

def mobilenet_ssd(image_size,
                  n_classes,
                  mode='training',
                  width_multiplier=1.0,
                  weights='imagenet',
                  mobilenet_version='v2',
                  l2_regularization=0.0005,
                  min_scale=None,
                  max_scale=None,
                  scales=None,
                  aspect_ratios_global=None,
                  aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                           [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                           [1.0, 2.0, 0.5],
                                           [1.0, 2.0, 0.5]],
                  two_boxes_for_ar1=True,
                  steps=[8, 16, 32, 64, 100, 300],
                  offsets=None,
                  clip_boxes=False,
                  variances=[0.1, 0.1, 0.2, 0.2],
                  coords='centroids',
                  normalize_coords=True,
                  subtract_mean=[128, 128, 128],
                  divide_by_stddev=[128, 128, 128],
                  swap_channels=False,
                  confidence_thresh=0.01,
                  iou_threshold=0.45,
                  top_k=200,
                  nms_max_output_size=400,
                  return_predictor_sizes=False):
    '''
    Build a Keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.

    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
            Set to zero to deactivate L2-regularization.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.

    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    img_shape = (img_height, img_width, img_channels)

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x0 = Input(shape=img_shape)

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=img_shape, name='identity_layer')(x0)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=img_shape, name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=img_shape, name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=img_shape, name='input_channel_swap')(x1)

    if mobilenet_version == 'v1':
        mobilenet_model = mobilenet.MobileNet(input_shape=img_shape, alpha=width_multiplier, weights=weights, include_top=False)
    elif mobilenet_version == 'v2':
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=img_shape, alpha=width_multiplier, weights=weights, include_top=False)

    def make_mobilenet_v1_layer(dw_kernel_size, dw_strides, dw_padding, pw_filters, input_tensor, layer_name):
        x = DepthwiseConv2D(dw_kernel_size, strides=dw_strides, activation=None, use_bias=False, padding=dw_padding, name='conv_dw_'+layer_name)(input_tensor)
        x = BatchNormalization(name='conv_dw_'+layer_name+'_bn')(x)
        x = Activation(mobilenet.relu6, name='conv_dw_'+layer_name+'_relu')(x)
        x = Conv2D(pw_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name='conv_'+layer_name+'_pw')(x)
        x = BatchNormalization(name='conv_pw_'+layer_name+'_bn')(x)
        x = Activation(mobilenet.relu6, name='conv_pw_'+layer_name+'_relu')(x)
        return x

    def make_mobilenet_v2_layer(dw_kernel_size, dw_strides, dw_padding, pw_filters, input_tensor, layer_name):
        x = DepthwiseConv2D(dw_kernel_size, strides=dw_strides, activation=None, use_bias=False, padding=dw_padding, name='block_'+layer_name+'_expand')(input_tensor)
        x = BatchNormalization(name='block_'+layer_name+'_expand_BN')(x)
        x = Activation(mobilenet.relu6, name='block_'+layer_name+'_expand_relu')(x)
        x = Conv2D(pw_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name='block_'+layer_name+'_depthwise')(x)
        x = BatchNormalization(name='block_'+layer_name+'_depthwise_BN')(x)
        x = Activation(mobilenet.relu6, name='block_'+layer_name+'_depthwise_relu')(x)
        return x

    def make_mobilenet_v2_ssdlite_layer(dw_kernel_size, dw_strides, dw_padding, pw_filters, input_tensor, layer_name):
        x = DepthwiseConv2D(dw_kernel_size, strides=dw_strides, activation=None, use_bias=False, padding=dw_padding, name=layer_name+'_expand')(input_tensor)
        x = BatchNormalization(name=layer_name+'_expand_BN')(x)
        x = Activation(mobilenet.relu6, name=layer_name+'_expand_relu')(x)
        x = Conv2D(pw_filters, kernel_size=1, padding='same', use_bias=True, activation=None, name=layer_name+'_depthwise')(x)
        x = Lambda(identity_layer, name=layer_name)(x)
        # x = BatchNormalization(name=layer_name+'_depthwise_BN')(x)
        # x = Activation(mobilenet.relu6, name='block_'+layer_name+'_depthwise_relu')(x)
        return x

    if mobilenet_version == 'v1':
        layer_11 = mobilenet_model.get_layer('conv_pw_11_relu').output
        layer_13 = mobilenet_model.get_layer('conv_pw_13_relu').output
        make_mobilenet_layer = make_mobilenet_v1_layer
    elif mobilenet_version == 'v2':
        layer_11 = mobilenet_model.get_layer('block_11_add').output
        layer_13 = mobilenet_model.get_layer('block_13_project_BN').output
        make_mobilenet_layer = make_mobilenet_v2_layer

    layer_14 = make_mobilenet_layer(3, (1,1), 'valid', int(width_multiplier*512), layer_13, '14')
    layer_15 = make_mobilenet_layer(3, (2,2), 'same', int(width_multiplier*256), layer_14, '15')
    layer_16 = make_mobilenet_layer(3, (2,2), 'same', int(width_multiplier*256), layer_15, '16')
    layer_17 = make_mobilenet_layer(2, (1,1), 'valid', int(width_multiplier*128), layer_16, '17')

    ### Build the convolutional predictor layers on top of the base network

    # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
    if mobilenet_version == 'v1':
        classes_11 = Conv2D(n_boxes[0]*n_classes, (3,3), strides=(1,1), padding="same", name='classes_11')(layer_11)
        classes_13 = Conv2D(n_boxes[1]*n_classes, (3,3), strides=(1,1), padding="same", name='classes_13')(layer_13)
        classes_14 = Conv2D(n_boxes[2]*n_classes, (3,3), strides=(1,1), padding="same", name='classes_14')(layer_14)
        classes_15 = Conv2D(n_boxes[3]*n_classes, (3,3), strides=(1,1), padding="same", name='classes_15')(layer_15)
        classes_16 = Conv2D(n_boxes[4]*n_classes, (3,3), strides=(1,1), padding="same", name='classes_16')(layer_16)
        classes_17 = Conv2D(n_boxes[5]*n_classes, (3,3), strides=(1,1), padding="same", name='classes_17')(layer_17)
    elif mobilenet_version == 'v2': # ssdlite
        classes_11 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[0]*n_classes, layer_11, 'classes_11')
        classes_13 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[1]*n_classes, layer_13, 'classes_13')
        classes_14 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[2]*n_classes, layer_14, 'classes_14')
        classes_15 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[3]*n_classes, layer_15, 'classes_15')
        classes_16 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[4]*n_classes, layer_16, 'classes_16')
        classes_17 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[5]*n_classes, layer_17, 'classes_17')

    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    if mobilenet_version == 'v1':
        boxes_11 = Conv2D(n_boxes[0]*4, (3,3), strides=(1,1), padding="same", name='boxes_11')(layer_11)
        boxes_13 = Conv2D(n_boxes[1]*4, (3,3), strides=(1,1), padding="same", name='boxes_13')(layer_13)
        boxes_14 = Conv2D(n_boxes[2]*4, (3,3), strides=(1,1), padding="same", name='boxes_14')(layer_14)
        boxes_15 = Conv2D(n_boxes[3]*4, (3,3), strides=(1,1), padding="same", name='boxes_15')(layer_15)
        boxes_16 = Conv2D(n_boxes[4]*4, (3,3), strides=(1,1), padding="same", name='boxes_16')(layer_16)
        boxes_17 = Conv2D(n_boxes[5]*4, (3,3), strides=(1,1), padding="same", name='boxes_17')(layer_17)
    elif mobilenet_version == 'v2': # ssdlite
        boxes_11 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[0]*4, layer_11, 'boxes_11')
        boxes_13 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[1]*4, layer_13, 'boxes_13')
        boxes_14 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[2]*4, layer_14, 'boxes_14')
        boxes_15 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[3]*4, layer_15, 'boxes_15')
        boxes_16 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[4]*4, layer_16, 'boxes_16')
        boxes_17 = make_mobilenet_v2_ssdlite_layer(3, (1,1), 'same', n_boxes[5]*4, layer_17, 'boxes_17')

    ## Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    anchors_11 = AnchorBoxes(
        img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
        clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors_11')(boxes_11)
    anchors_13 = AnchorBoxes(
        img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
        clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors_13')(boxes_13)
    anchors_14 = AnchorBoxes(
        img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
        clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors_14')(boxes_14)
    anchors_15 = AnchorBoxes(
        img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
        clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors_15')(boxes_15)
    anchors_16 = AnchorBoxes(
        img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4],
        clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors_16')(boxes_16)
    anchors_17 = AnchorBoxes(
        img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5],
        clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors_17')(boxes_17)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    classes_11_reshaped = Reshape((-1, n_classes), name='classes_11_reshape')(classes_11)
    classes_13_reshaped = Reshape((-1, n_classes), name='classes_13_reshape')(classes_13)
    classes_14_reshaped = Reshape((-1, n_classes), name='classes_14_reshape')(classes_14)
    classes_15_reshaped = Reshape((-1, n_classes), name='classes_15_reshape')(classes_15)
    classes_16_reshaped = Reshape((-1, n_classes), name='classes_16_reshape')(classes_16)
    classes_17_reshaped = Reshape((-1, n_classes), name='classes_17_reshape')(classes_17)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes_11_reshaped = Reshape((-1, 4), name='boxes_11_reshape')(boxes_11)
    boxes_13_reshaped = Reshape((-1, 4), name='boxes_13_reshape')(boxes_13)
    boxes_14_reshaped = Reshape((-1, 4), name='boxes_14_reshape')(boxes_14)
    boxes_15_reshaped = Reshape((-1, 4), name='boxes_15_reshape')(boxes_15)
    boxes_16_reshaped = Reshape((-1, 4), name='boxes_16_reshape')(boxes_16)
    boxes_17_reshaped = Reshape((-1, 4), name='boxes_17_reshape')(boxes_17)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors_11_reshaped = Reshape((-1, 8), name='anchors_11_reshape')(anchors_11)
    anchors_13_reshaped = Reshape((-1, 8), name='anchors_13_reshape')(anchors_13)
    anchors_14_reshaped = Reshape((-1, 8), name='anchors_14_reshape')(anchors_14)
    anchors_15_reshaped = Reshape((-1, 8), name='anchors_15_reshape')(anchors_15)
    anchors_16_reshaped = Reshape((-1, 8), name='anchors_16_reshape')(anchors_16)
    anchors_17_reshaped = Reshape((-1, 8), name='anchors_17_reshape')(anchors_17)


    ### Concatenate the predictions from the different layers

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([
        classes_11_reshaped,
        classes_13_reshaped,
        classes_14_reshaped,
        classes_15_reshaped,
        classes_16_reshaped,
        classes_17_reshaped
    ])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([
        boxes_11_reshaped,
        boxes_13_reshaped,
        boxes_14_reshaped,
        boxes_15_reshaped,
        boxes_16_reshaped,
        boxes_17_reshaped
    ])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([
        anchors_11_reshaped,
        anchors_13_reshaped,
        anchors_14_reshaped,
        anchors_15_reshaped,
        anchors_16_reshaped,
        anchors_17_reshaped
    ])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

    if mode == 'training':
        model = Model(x0, Model(mobilenet_model.input, predictions)(x1))
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(x0, Model(mobilenet_model.input, decoded_predictions)(x1))
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(x0, Model(mobilenet_model.input, decoded_predictions)(x1))
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([classes_11._keras_shape[1:3],
                                    classes_13._keras_shape[1:3],
                                    classes_14._keras_shape[1:3],
                                    classes_15._keras_shape[1:3],
                                    classes_16._keras_shape[1:3],
                                    classes_17._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model

