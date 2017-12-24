"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from .. import initializers
from .. import layers
from .. import losses

import numpy as np

custom_objects = {
    'UpsampleLike'          : layers.UpsampleLike,
    'PriorProbability'      : initializers.PriorProbability,
    'RegressBoxes'          : layers.RegressBoxes,
    'NonMaximumSuppression' : layers.NonMaximumSuppression,
    'Anchors'               : layers.Anchors,
    '_smooth_l1'            : losses.smooth_l1(),
    '_focal'                : losses.focal(),
}


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """Returns a Keras model for the classification subnet of
    RetinaNet. This model is a 5-layer FCN whose inputs is a feature
    map from any level of the feature pyramid, and whose output is the
    probability of each class and each anchor.
   
    Parameters:
    num_classes -- Number of object classes
    num_anchors -- Number of anchors at each sliding-window location
    pyramid_feature_size -- Dimensions of feature maps (default 256)
    prior_probability -- Prior probability of each object class (default 0.01)
    classification_feature_size -- Dimensions of each hidden layer (default 256)
    name -- Network name (default 'classification_submodel')
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    # Input is a feature map:
    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    # Pass 'input' as the first layer's input, that layer's output to
    # the input of the next, and so on, to produce the first 4 layers
    # (note the initializer; see p5 of the RetinaNet paper):
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    # Produce the final convolutional layer (note different
    # initialization and dimensions; see RetinaNet figure 3):
    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(
    num_anchors,
    pyramid_feature_size=256,
    regression_feature_size=256,
    name='regression_submodel'
): 
    """Returns a Keras model for the box regression subnet of
    RetinaNet. As in the classification subnet, this model is a
    5-layer FCN whose inputs is a feature map from any level of the
    feature pyramid. Its output is (for each anchor) a 4-coordinate
    parametrization giving a box relative to the anchor box.
   
    Parameters:
    num_anchors -- Number of anchors at each sliding-window location
    pyramid_feature_size -- Dimensions of feature maps (default 256)
    classification_feature_size -- Dimensions of each hidden layer (default 256)
    name -- Network name (default 'regression_submodel')
    """
   
    # "All new conv layers except the final one in the RetinaNet
    # subnets are initialized with bias b = 0 and a Gaussian weight
    # fill with stddev = 0.01."
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    # This works near-identically to default_classification_model:
    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """Returns the feature pyramid as 5 Keras layers, (P3,P4,P5,P6,P7).
    See p4 of the RetinaNet paper for an explanation of these layers.

    Parameters:
    C3 -- Final layer of 3rd residual stage of ResNet
    C3 -- Final layer of 4th residual stage of ResNet
    C3 -- Final layer of 5th residual stage of ResNet
    feature_size -- Dimensions of feature space (default 256)
    """
    
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7


class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    # See RetinaNet paper p4 (areas are 32^2 to 512^2):
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def default_submodels(num_classes, anchor_parameters):
    """Returns a list of (name, model) for the submodels in this network.

    """
    return [
        ('regression', default_regression_model(anchor_parameters.num_anchors())),
        ('classification', default_classification_model(num_classes, anchor_parameters.num_anchors()))
    ]


def __build_model_pyramid(name, model, features):
    """Returns Keras layer containing the concatenated results of a model
    applied to several sets of features.

    Parameters:
    name -- Name of returned layer
    model -- Keras model which should operate on 'features' elements
    features -- List of Keras layers
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """Returns a list of Keras layers, with each layer containing one
    model from 'models' applied to each layer in 'features'.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """Returns a Keras layer containing all anchors, concatenated for each
    element of 'features' (which should correspond with each element
    of 'anchor_parameters'
    """
    anchors = []
    for i, _ in enumerate(features):
        anchors.append(layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f))
    return keras.layers.Concatenate(axis=1)(anchors)


def retinanet(
    inputs,
    backbone,
    num_classes,
    anchor_parameters       = AnchorParameters.default,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    """Returns a Keras model with the classification and box regression
    subnet, as well as the anchor bounding boxes.  Model outputs are:
    [anchors, regression subnet output, classification subnet output].

    Parameters:
    inputs -- Input layer used for network
    backbone -- Backbone network whose outputs are used for feature pyramid
    num_classes -- How many classes to use for classification subnet
    anchor_parameters -- Optional AnchorParameters instance for anchors
    create_pyramid_features -- Optional function for building pyramid;
                               should take arguments for C3, C4, and C5
                               and returns a tuple with (P3,P4,P5,P6,P7).
    name -- Name of model (default "retinanet")
    """
    if submodels is None:
        submodels = default_submodels(num_classes, anchor_parameters)

    _, C3, C4, C5 = backbone.outputs  # we ignore C2

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramid = __build_pyramid(submodels, features)
    anchors = __build_anchors(anchor_parameters, features)

    return keras.models.Model(inputs=inputs, outputs=[anchors] + pyramid, name=name)


def retinanet_bbox(inputs, num_classes, nms=True, name='retinanet-bbox', *args, **kwargs):
    model = retinanet(inputs=inputs, num_classes=num_classes, *args, **kwargs)

    # we expect the anchors, regression and classification values as first output
    anchors        = model.outputs[0]
    regression     = model.outputs[1]
    classification = model.outputs[2]

    # apply predicted regression to anchors
    boxes      = layers.RegressBoxes(name='boxes')([anchors, regression])
    detections = keras.layers.Concatenate(axis=2)([boxes, classification] + model.outputs[3:])

    # additionally apply non maximum suppression
    if nms:
        detections = layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])

    # construct the model
    # (Note that anchors are ignored in outputs)
    return keras.models.Model(inputs=inputs, outputs=model.outputs[1:] + [detections], name=name)
