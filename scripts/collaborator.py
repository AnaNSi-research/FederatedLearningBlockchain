import os
import sys

# Get the directory containing this script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Add the directory to sys.path
sys.path.insert(0, dir_path)


import tensorflow as tf

"""
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""
        
from brownie import FederatedLearning, network, config, accounts
import ipfshttpclient

import numpy as np
import json
import time
import io
from threading import Timer
import asyncio

"""
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
# from multiprocessing import Pool
"""

from utils_simulation import get_hospitals
from utils_collaborator import NUM_EPOCHS

import tracemalloc
tracemalloc.start()

hospitals = get_hospitals()

for hospital in hospitals:
    print(hospital)

evalu = {}


client = ipfshttpclient.connect()
federated_learning = FederatedLearning[-1]
# account = accounts[1]

contractEvents = federated_learning.events
contractEvent = network.contract.ContractEvents(network.contract.Contract(federated_learning.address))

async def changeState_alert(event):
    print("LOLOLOLOLOLOLO", event["args"]["new_state"])
    #print("LOLOLOLOLOLOLO", event)
    match event["args"]["new_state"]:
    #match event:
        case "OPEN":
            print("OPEN")
        case "START":
            print("START")
            start_event()
        case "LEARNING":
            print("LEARNING")
            learning_event()
        case "CLOSE":
            print("CLOSE")
            close_event()
        case _:
            print("ERROR")
    print(event)


async def aggregatedWeightsReady_alert(event):
    for hospital_name in hospitals:
        retrieving_aggreagted_weights(hospital_name, federated_learning, client)
        fitting(hospital_name)
        loading_weights(hospital_name, federated_learning, client)

        # await for AW
        coroutine_AW = contractEvent.listen("AggregatedWeightsReady")
        print(coroutine_AW)
        print(type(coroutine_AW))
        await coroutine_AW

        """
        # await for CLOSE
        coroutine_close = contractEvent.listen("ChangeState")
        print(coroutine_close)
        print(type(coroutine_close))
        await coroutine_close
        """


def start_event():
    print(hospitals)
    """ The collaborators download model and compile information from the blockchain """
    for hospital_name in hospitals:
        # model
        retrieve_model_tx = federated_learning.retrieve_model(
            {"from": hospitals[hospital_name].address}
        )
        retrieve_model_tx.wait(1)

        retrieved_model = retrieve_model_tx.return_value
        decoded_model = retrieved_model.decode("utf-8")
        model = tf.keras.models.model_from_json(decoded_model)
        hospitals[hospital_name].model = model

        # compile_info
        retreive_compile_info_tx = federated_learning.retrieve_compile_info(
            {"from": hospitals[hospital_name].address}
        )
        retreive_compile_info_tx.wait(1)
        
        retreived_compile_info = retreive_compile_info_tx.return_value
        decoded_compile_info = retreived_compile_info.decode("utf-8")
        fl_compile_info = json.loads(decoded_compile_info)
        hospitals[hospital_name].compile_info = fl_compile_info

        hospitals[hospital_name].model.compile(**hospitals[hospital_name].compile_info)


def learning_event():
    """ Execution of fedederating learning rounds """
    for hospital_name in hospitals:
        fitting(hospital_name)
        loading_weights(hospital_name, federated_learning, client)
        

async def close_event():
    pass



def fitting(_hospital_name):
    hospitals[_hospital_name].model.fit(
        hospitals[_hospital_name].dataset["X_train"],
        hospitals[_hospital_name].dataset["y_train"],
        validation_data=(
            hospitals[_hospital_name].dataset["X_val"],
            hospitals[_hospital_name].dataset["y_val"],
        ),
        epochs=NUM_EPOCHS,
    )

    evalu[_hospital_name].append(
        hospitals[_hospital_name].model.evaluate(
            hospitals[_hospital_name].dataset["X_test"],
            hospitals[_hospital_name].dataset["y_test"],
        )
    )

    hospitals[_hospital_name].weights = hospitals[_hospital_name].model.get_weights()


def loading_weights(_hospital_name, _fl_contract, client):
    weights = hospitals[_hospital_name].weights

    # print_weights(weights)

    weights_listed = [param.tolist() for param in weights]
    # weights_listed = weights_listed[:3]

    # print_listed_weights(weights_listed)

    weights_JSON = json.dumps(weights_listed)
    print("weights_JSON size:" + str(sys.getsizeof(weights_JSON)))

    start_time = time.time()
    res = client.add(io.BytesIO(weights_JSON.encode("utf-8")))
    print("ADD TIME: ", str(time.time() - start_time))
    print("RES: ", res.keys())

    print(
        _fl_contract.send_weights(
            res["Hash"].encode("utf-8"),
            {"from": hospitals[_hospital_name].address},
        )
    )

def retrieving_aggreagted_weights(_hospital_name, _fl_contract, client):
    weight_hash = _fl_contract.retrieve_aggregated_weights(
        {"from": hospitals[_hospital_name].address}
    )

    weight_hash = weight_hash.decode("utf-8")
    start_time = time.time()
    w_get = client.cat(weight_hash)
    print("GET TIME: ", str(time.time() - start_time))
    w_get = w_get.decode("utf-8")
    w_listed_worker = json.loads(w_get)
    w_worker = [np.array(param, dtype=np.float32) for param in w_listed_worker]

    # print_weights(w_worker)

    hospitals[_hospital_name].aggregated_weights = w_worker
    hospitals[_hospital_name].model.set_weights(w_worker)


async def main():
    
    # only with fl-local network
    for idx, hospital_name in enumerate(hospitals, start=1):
        hospitals[hospital_name].address = accounts[idx]
        print(idx, hospitals[hospital_name].address)

    #contractEvent.subscribe("ChangeState", changeState_alert, delay=0.5)
    #contractEvent.subscribe("AggregatedWeightsReady", aggregatedWeightsReady_alert, delay=0.5)

    # await for OPEN
    coroutine_open = contractEvent.listen("ChangeState")
    print("COROUTINE: waiting OPEN\n", coroutine_open)
    await coroutine_open
    print("I waited OPEN\n_____________________________________")
    
    #await for START
    coroutine_start = contractEvent.listen("ChangeState")
    print("COROUTINE: waiting START\n", coroutine_start)
    await coroutine_start
    print("I waited START\n_____________________________________")
    start_event()
    
    #await for LEARNING
    coroutine_learning = contractEvent.listen("ChangeState")
    print("COROUTINE: waiting LEARNING\n", coroutine_learning)
    await coroutine_learning
    print("I waited LEARNING\n_____________________________________")
    learning_event()




    print("END")



asyncio.run(main())
