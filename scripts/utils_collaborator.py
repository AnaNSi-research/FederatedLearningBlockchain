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
import sys
import os
import cv2

NUM_EPOCHS = 10


