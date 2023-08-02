from utils_simulation import RANDOM_SEED, set_reproducibility

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
    Rescaling,
)
import numpy as np
import json
import io

WIDTH = 176
HEIGHT = 208
DEPTH = 1
NUM_CLASSES = 4

NUM_ROUNDS = 5

TIMEOUT_SECONDS = 600
EPSILON = 10 ** (-5)

# similarity = ['single', 'multiple', 'averaged']
SIMILARITY = "single"

compile_info = {
    "loss": "CategoricalCrossentropy",
    "optimizer": "Adam",
    "metrics": ["accuracy"],
}


def create_model(input_shape, num_classes):
    set_reproducibility(RANDOM_SEED)
    glorot_initializer = GlorotUniform(seed=RANDOM_SEED)

    x = Input(shape=input_shape)

    r = Rescaling(1.0 / 255)(x)

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
    do1 = Dropout(0.3, seed=RANDOM_SEED)(d1)
    d2 = Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=glorot_initializer,
    )(do1)

    model = Model(x, d2)

    return model


def get_encoded_compile_info():
    JSON_compile_info = json.dumps(compile_info)
    encoded_compile_info = JSON_compile_info.encode("utf-8")
    return encoded_compile_info


def get_encoded_model(input_shape, num_classes):
    model = create_model(input_shape, num_classes)
    JSON_model = model.to_json()
    encoded_model = JSON_model.encode("utf-8")
    return encoded_model


def assert_coroutine_result(_coroutine_result, _function_name):
    if _coroutine_result.event_data.args.functionName == _function_name:
        print(f'The event "{_function_name}" has been correctly catched')
    else:
        raise Exception('ERROR: event "', _function_name, '" not catched')


def weights_encoding(_weights):
    weights_listed = [param.tolist() for param in _weights]
    weights_JSON = json.dumps(weights_listed)
    weights_encoded = weights_JSON.encode("utf-8")
    weights_bytes = io.BytesIO(weights_encoded)
    return weights_bytes


def weights_decoding(_weights_encoded):
    weights_JSON = _weights_encoded.decode("utf-8")
    weights_listed = json.loads(weights_JSON)
    weights = [np.array(param, dtype=np.float32) for param in weights_listed]
    return weights


def similarity_single(
    _hospital_address, _hospitals_weights, _averaged_weights, _hospitals_addresses
):
    numerator = [
        np.linalg.norm(h_w - a_w)
        for hospital_address in _hospitals_addresses
        for h_w, a_w in zip(_hospitals_weights[hospital_address], _averaged_weights)
    ]
    numerator = sum(numerator)

    denominator = [
        np.linalg.norm(h_w - a_w)
        for h_w, a_w in zip(_hospitals_weights[_hospital_address], _averaged_weights)
    ]
    denominator = sum(denominator) + (10**-5)

    result = numerator / denominator
    return result



def similarity_factor_single(
    _hospital_address, _hospital_weights, _averaged_weights, _hospitals_addresses
):
    return similarity_single(
        _hospital_address, _hospital_weights, _averaged_weights, _hospitals_addresses
    ) / sum(
        [
            similarity_single(
                hospital_address,
                _hospital_weights,
                _averaged_weights,
                _hospitals_addresses,
            )
            for hospital_address in _hospitals_addresses
        ]
    )



# utility function to compute the Frobenius norm between 2 matrices
def frobenius_norm(_hospital_address, _hospitals_weights, _averaged_weights):
    result = [
        np.linalg.norm(h_w - a_w)
        for h_w, a_w in zip(_hospitals_weights[_hospital_address], _averaged_weights)
    ]
    return np.array(result)


# utility function to compute the similarity factor used in the aggregated weights
def similarity_multiple(
    _hospital_address, _hospitals_weights, _averaged_weights, _hospitals_addresses
):
    distances = [
        frobenius_norm(hospital_address, _hospitals_weights, _averaged_weights)
        for hospital_address in _hospitals_addresses
    ]

    numerator = [sum(layer) for layer in zip(*distances)]
    denominator = (
        frobenius_norm(_hospital_address, _hospitals_weights, _averaged_weights)
        + EPSILON
    )

    result = np.divide(numerator, denominator)
    return result


# return the weighted contribute of a single collaborator on a FL round
def similarity_factor_multiple(
    _hospital_address, _hospitals_weights, _averaged_weights, _hospitals_addresses
):
    similarities = [
        similarity_multiple(
            hospital_address,
            _hospitals_weights,
            _averaged_weights,
            _hospitals_addresses,
        )
        for hospital_address in _hospitals_addresses
    ]

    numerator = similarity_multiple(
        _hospital_address, _hospitals_weights, _averaged_weights, _hospitals_addresses
    )

    denominator = [sum(layer) for layer in zip(*similarities)]

    result = np.divide(numerator, denominator)
    return result




