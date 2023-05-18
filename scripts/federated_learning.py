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
from web3 import Web3, HTTPProvider
import asyncio

"""
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
# from multiprocessing import Pool
"""

from helpful_scripts import get_account
from utils_manager import *

client = ipfshttpclient.connect()
federated_learning = FederatedLearning[-1]
manager = get_account()

contractEvents = federated_learning.events
contractEvent = network.contract.ContractEvents(network.contract.Contract(federated_learning.address))

"""
eventWatcher = network.event.EventWatcher()
blockchain_address = 'http://127.0.0.1:7545'
web3 = Web3(HTTPProvider(blockchain_address))
web3.eth.defaultAccount = get_account()
contract_address = federated_learning.address
contract_abi = federated_learning.abi
web3_contract = web3.eth.contract(address=contract_address, abi=contract_abi)
# web3_contract = web3.contract.Contract(contract_address)
"""

timers = {}
m_evalu = {}
round = 0

def startTimeout(phase):
    print("PRE_START", timers)
    timers[phase] = Timer(TIMEOUT_SECONDS, timeoutExpired_alert, args=[phase])
    timers[phase].start()
    print("POST_START", timers)

def deleteTimeout(phase):
    print("PRE_DELETE", timers)
    timers[phase].cancel()
    timers.pop(phase)
    print("POST_DELETE", timers)


async def everyCollaboratorhasCalledOnlyOnce_alert(event):
    print("EVERYEVERYEVERYEVERY")
    match event["args"]["functionName"]:
        case "retrieve_model":
            # await for retrieve_compile_info
            coroutine_RCI = contractEvent.listen("EveryCollaboratorhasCalledOnlyOnce")
            print(coroutine_RCI)
            print(type(coroutine_RCI))
            await coroutine_RCI
        case "retrieve_compile_info":
            deleteTimeout("start")
            startTimeout("learning")
            """ Starting the federating learning """
            # print(federated_learning.get_state())
            learning_tx = federated_learning.learning({"from": manager})
            learning_tx.wait(1)
            print(learning_tx.events)
            # print(federated_learning.get_state())

            print("LOL")
            # await for send_weights
            coroutine_SW = contractEvent.listen("EveryCollaboratorhasCalledOnlyOnce")
            print(coroutine_SW)
            print(type(coroutine_SW))
            await coroutine_SW
            print("LUL")

        case "send_weights":
            if round < NUM_ROUNDS:
                manager_weights(federated_learning, manager, client)
                # await for retrieve_aggregated_weights
                coroutine_RAW= contractEvent.listen("EveryCollaboratorhasCalledOnlyOnce")
                print(coroutine_RAW)
                print(type(coroutine_RAW))
                await coroutine_RAW
            else:
                # print(federated_learning.get_state())
                startTimeout("close")
                close_tx = federated_learning.close({"from": manager})
                close_tx.wait(1)
                print(close_tx.events)
                # print(federated_learning.get_state())
        case "retrieve_aggregated_weights":
            # è VERAMENTE NECESSARIO GESTIRLO?

            # await for send_weights
            coroutine_SW = contractEvent.listen("EveryCollaboratorhasCalledOnlyOnce")
            print(coroutine_SW)
            print(type(coroutine_SW))
            await coroutine_SW
        case _:
            print("ERROR")


def timeoutExpired_alert(phase):
    match phase:
        case "start":
            """ Starting the federating learning """
            # print(federated_learning.get_state())
            learning_tx = federated_learning.learning({"from": manager})
            learning_tx.wait(1)
            print(learning_tx.events)
            # print(federated_learning.get_state())
            startTimeout("learning")
        case "learning":
            manager_weights(federated_learning, manager, client)
        case "send_aggregated_weights":
            manager_weights(federated_learning, manager, client)
        case _:
            print("ERROR")

def manager_weights(_fl_contract, _manager, client):
    retreived_weights_hash = _fl_contract.retrieve_weights({"from": _manager})
    hospitals_weights = []
    len_h_w = 0
    for weight_hash in retreived_weights_hash:
        weight_hash = weight_hash.decode("utf-8")
        start_time = time.time()
        w_get = client.cat(weight_hash)
        print("GET TIME: ", str(time.time() - start_time))
        w_get = w_get.decode("utf-8")
        w_listed_worker = json.loads(w_get)
        w_worker = [np.array(param, dtype=np.float32) for param in w_listed_worker]


        hospitals_weights.append(w_worker)
    len_h_w = len(hospitals_weights)
    print("END")

    # aggregation
    averaged_weights = []

    for i in range(len(hospitals_weights[0])):
        layer_weights = []
        for j in range(len_h_w):
            layer_weights.append(hospitals_weights[j][i])
        averaged_weights.append(sum(layer_weights) / len(hospitals_weights))

    for i in range(len(averaged_weights)):
        averaged_weights[i] = np.array(
            averaged_weights[i]
        )  # Convert the list to a NumPy array

    def similarity(_hospital_idx):
        numerator = [
            np.linalg.norm(arr)
            for weight in hospitals_weights
            for arr in np.subtract(weight, averaged_weights)
        ]
        numerator = sum(numerator)
        print("NUMERATOR: ", numerator)

        denominator = [
            np.linalg.norm(arr)
            for arr in np.subtract(hospitals_weights[_hospital_idx], averaged_weights)
        ]
        denominator = sum(denominator) + (10**-5)
        print("DENOMINATOR: ", denominator)

        result = numerator / denominator
        return result

    def similarity_factor(_hospital_idx):
        return similarity(_hospital_idx) / sum(
            [similarity(hospital_idx) for hospital_idx in range(len_h_w)]
        )

    factors = {}
    for hospital_idx in range(len_h_w):
        factors[hospital_idx] = similarity_factor(hospital_idx)
    print("FACTORS: ", str(type(factors)), str(len(factors)), str(factors))

    aggregated_weights = []

    for i in range(len(hospitals_weights[0])):
        layer_weights = []
        for hospital_idx in range(len_h_w):
            layer_weights.append(
                factors[hospital_idx] * hospitals_weights[hospital_idx][i]
            )
        aggregated_weights.append(sum(layer_weights))

    for i in range(len(aggregated_weights)):
        aggregated_weights[i] = np.array(
            aggregated_weights[i]
        )  # Convert the list to a NumPy array

    # aggregated_weights = averaged_weights

    model_agg = create_model((HEIGHT, WIDTH, DEPTH), NUM_CLASSES)
    model_agg.compile(**compile_info)
    model_agg.set_weights(aggregated_weights)
    m_evalu.append(model_agg.evaluate(get_X_test(), get_y_test()))

    weights_listed = [param.tolist() for param in aggregated_weights]
    weights_JSON = json.dumps(weights_listed)

    res = client.add(io.BytesIO(weights_JSON.encode("utf-8")))
    print(
        _fl_contract.send_aggregated_weights(
            res["Hash"].encode("utf-8"), {"from": _manager}
        )
    )
    startTimeout("send_aggregated_weights")

async def main():
    # client = ipfshttpclient.connect()
    # manager = get_account()
    # federated_learning = FederatedLearning[-1]

    #contractEvent.subscribe("ChangeState", changeState_alert, delay=0.5)
    contractEvent.subscribe("EveryCollaboratorhasCalledOnlyOnce", everyCollaboratorhasCalledOnlyOnce_alert, delay=0.5)
    #contractEvent.subscribe("AggregatedWeightsReady", aggregatedWeightsReady_alert, delay=0.5)
    #contractEvents.subscribe("TimeoutExpired", timeoutExpired_alert, delay=0.0)

    """
    eventWatcher.add_event_callback(web3_contract.events.ChangeState(), changeState_alert, delay=0.5)
    eventWatcher.add_event_callback(web3_contract.events.EveryCollaboratorhasCalledOnlyOnce(), everyCollaboratorhasCalledOnlyOnce_alert, delay=0.5)
    eventWatcher.add_event_callback(web3_contract.events.AggregatedWeightsReady(), aggregatedWeightsReady_alert, delay=0.5)
    """

    # model = create_model((HEIGHT, WIDTH, DEPTH), NUM_CLASSES)

    """ The manager uploads model and compile information on the blockchain """
    encoded_model = get_encoded_model((HEIGHT, WIDTH, DEPTH), NUM_CLASSES)
    # print(federated_learning.get_model())
    send_model_tx = federated_learning.send_model(encoded_model, {"from": manager})
    send_model_tx.wait(1)
    print(send_model_tx.events)
    # print(federated_learning.get_model())

    encoded_compile_info = get_encoded_compile_info()
    # print(federated_learning.get_compile_info())
    send_compile_info_tx = federated_learning.send_compile_info(encoded_compile_info, {"from": manager})
    send_compile_info_tx.wait(1)
    print(send_compile_info_tx.events)
    # print(federated_learning.get_compile_info())

    """ Retrieving of the model """
    # print(federated_learning.get_state())
    startTimeout("start")
    start_tx = federated_learning.start({"from": manager})
    start_tx.wait(1)
    print(start_tx.events)
    # print(federated_learning.get_state())

    # await for retrieve_model
    coroutine_RM = contractEvent.listen("EveryCollaboratorhasCalledOnlyOnce")
    print(coroutine_RM)
    print(type(coroutine_RM))
    await coroutine_RM

    # time.sleep(120)
    print("END")

    """
    

    # EVENTO DA PARTE DEI COLLABORATORI: IL LEARNING PUò INIZIARE

    #Closing the blockchain 
    # print(federated_learning.get_state())
    close_tx = federated_learning.close({"from": manager})
    close_tx.wait(1)
    print(close_tx.events)
    # print(federated_learning.get_state())


    print("HOSPITAL_EVAL:")
    for h in evalu:
        print(evalu[h])
        print("----------------------------")
    print("MANAGER_EVAL:", m_evalu)

    """

asyncio.run(main())
