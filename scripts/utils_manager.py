import sys
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import GlorotUniform, Zeros
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Rescaling
)
import numpy as np
import json
from utils_simulation import RANDOM_SEED
from hospitals import get_X_test, get_y_test


WIDTH = 176
HEIGHT = 208
DEPTH = 1
NUM_CLASSES = 4
NUM_EPOCHS = 10

NUM_ROUNDS = 5

TIMEOUT_SECONDS = 600


compile_info = {
    "loss": "SparseCategoricalCrossentropy",
    "optimizer": "Adam",
    "metrics": ["accuracy"],
}


def create_model(input_shape, num_classes):
    glorot_initializer = GlorotUniform(seed=RANDOM_SEED)

    x = Input(shape=input_shape)

    r = Rescaling(1./255)(x)

    c1 = Conv2D(
        16,
        3,
        padding="same",
        activation="relu",
        kernel_initializer=glorot_initializer,
    )(r)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        kernel_initializer=glorot_initializer,
    )(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_initializer=glorot_initializer,
    )(p2)
    p3 = MaxPooling2D((2, 2))(c3)

    f = Flatten()(p3)
    d1 = Dense(
        128,
        activation="relu",
        kernel_initializer=glorot_initializer,
        kernel_constraint=MaxNorm(3),
    )(f)
    do1 = Dropout(0.3)(d1)
    d2 = Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=glorot_initializer,
    )(do1)

    model = Model(x, d2)

    # model.summary()

    return model


def get_encoded_compile_info():
    JSON_compile_info = json.dumps(compile_info)
    encoded_compile_info = JSON_compile_info.encode("utf-8")
    return encoded_compile_info


def get_encoded_model(input_shape, num_classes):
    model = create_model(input_shape, num_classes)
    # print("memory model:" + str(sys.getsizeof(model)))
    JSON_model = model.to_json()
    # print("memory model_JSON:" + str(sys.getsizeof(global_model_JSON)))
    encoded_model = JSON_model.encode("utf-8")
    return encoded_model


# ALTERNATIVE MODEL
"""
    glorot_initializer = GlorotUniform(seed=RANDOM_SEED)
    from tf.keras.layers import BatchNormalization
    from tf.keras.regularizers import L2


    x = Input(shape=input_shape)

    # r = Rescaling(1./255)(x)

    c1 = Conv2D(
        16,
        3,
        padding="same",
        activation="relu",
        kernel_initializer=glorot_initializer,
        # kernel_regularizer=L2(),
    )(x)
    b1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(b1)
    b1 = BatchNormalization()(p1)
    c2 = Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        kernel_initializer=glorot_initializer,
        # kernel_regularizer=L2(),
    )(p1)
    b2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(b2)
    b2 = BatchNormalization()(p2)
    c3 = Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_initializer=glorot_initializer,
        kernel_regularizer=L2(),
    )(p2)
    b3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(b3)
    b3 = BatchNormalization()(p3)

    f = Flatten()(b3)
    bf = BatchNormalization()(f)
    d1 = Dense(
        128,
        activation="relu",
        kernel_initializer=glorot_initializer,
        kernel_regularizer=L2()
    )(bf)
    bd = BatchNormalization()(d1)
    d2 = Dense(
        NUM_CLASSES,
        activation="softmax",
        kernel_initializer=glorot_initializer,
        # kernel_regularizer=L2(),
    )(bd)
    bd = BatchNormalization()(d2)

    model = Model(x, bd)

    # model.summary()

    return model
"""
