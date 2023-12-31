import os
import sys

# Get the directory containing this script and add it to the sys.path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from utils_simulation import (
    RANDOM_SEED,
    PIN_BOOL,
    get_hospitals,
    print_line,
    set_reproducibility,
)
from utils_collaborator import *

from brownie import FederatedLearning
import ipfshttpclient

from tensorflow.keras.models import model_from_json
import asyncio
import json
import time


set_reproducibility(RANDOM_SEED)

# retrieve the hospitals information
hospitals = get_hospitals()

# connect to IPFS and Blockchain
IPFS_client = ipfshttpclient.connect()
FL_contract = FederatedLearning[-1]

# manage contract events
contract_events = FL_contract.events

# storing the hospitals performance results through the Federated Learning rounds
hospitals_evaluation = {hospital_name: [] for hospital_name in hospitals}


def closeState_alert(event):
    print("The FL Blockchain has been CLOSED\n")
    print("RESULTS - Hospitals Performance Evaluation through Federated Learning...")
    for hospital_name in hospitals_evaluation:
        print(f"{hospital_name}:")
        for round, [loss, acc] in enumerate(hospitals_evaluation[hospital_name], start=1):
            print(f"\tRound {round}:\tLoss: {loss:.3f} - Accuracy: {acc:.3f}")
    # network.disconnect()
    # sys.exit(0)


# triggered after the START event from the Blockchain
def start_event():
    for hospital_name in hospitals:

        # retrieving of the model given by the Manager
        retrieve_model_tx = FL_contract.retrieve_model(
            {"from": hospitals[hospital_name].address}
        )
        retrieve_model_tx.wait(1)

        decoded_model = decode_utf8(retrieve_model_tx)
        model = model_from_json(decoded_model)
        hospitals[hospital_name].model = model

        # retrieving of the compile information goven by the Manager
        retreive_compile_info_tx = FL_contract.retrieve_compile_info(
            {"from": hospitals[hospital_name].address}
        )
        retreive_compile_info_tx.wait(1)

        decoded_compile_info = decode_utf8(retreive_compile_info_tx)
        fl_compile_info = json.loads(decoded_compile_info)
        hospitals[hospital_name].compile_info = fl_compile_info

        # compiling the model with the compile information
        hospitals[hospital_name].model.compile(**hospitals[hospital_name].compile_info)


# operations to do at every FL round
def round_loop():
    for hospital_name in hospitals:
        fitting_model_and_loading_weights(hospital_name)


# triggered after the 'aggregatedWeightsReady' event from the Blockchain
def aggregatedWeightsReady_event():
    for hospital_name in hospitals:
        retrieving_aggreagted_weights(hospital_name)


def fitting_model_and_loading_weights(_hospital_name):

    """fitting the model"""
    X_train = hospitals[_hospital_name].dataset["X_train"]
    y_train = hospitals[_hospital_name].dataset["y_train"]
    X_val = hospitals[_hospital_name].dataset["X_val"]
    y_val = hospitals[_hospital_name].dataset["y_val"]
    X_test = hospitals[_hospital_name].dataset["X_test"]
    y_test = hospitals[_hospital_name].dataset["y_test"]

    # fitting the model with the aggreagted weights from the Manager
    hospitals[_hospital_name].model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
    )

    hospitals_evaluation[_hospital_name].append(
        hospitals[_hospital_name].model.evaluate(X_test, y_test)
    )

    # updating the old weights with the parameters from the newly fitted model
    hospitals[_hospital_name].weights = hospitals[_hospital_name].model.get_weights()

    """ loading weights """
    weights = hospitals[_hospital_name].weights
    weights_bytes = weights_encoding(weights)
    # print("weights_JSON size:" + str(sys.getsizeof(weights_JSON)))

    # uploading the weights on IPFS
    start_time = time.time()
    add_info = IPFS_client.add(weights_bytes, pin=PIN_BOOL)
    print("IPFS 'add' time: ", str(time.time() - start_time))
    print("IPFS 'add' info: ", add_info.keys())

    # sending the IPFS hash of the weights in the Blockchain
    hash_encoded = add_info["Hash"].encode("utf-8")
    send_weights_tx = FL_contract.send_weights(
        hash_encoded,
        {"from": hospitals[_hospital_name].address},
    )
    send_weights_tx.wait(1)


def retrieving_aggreagted_weights(_hospital_name):
    # retrieve the IPFS hash of the aggregated wights from the Blockchain
    retrieve_aggregated_weights_tx = FL_contract.retrieve_aggregated_weights(
        {"from": hospitals[_hospital_name].address}
    )
    retrieve_aggregated_weights_tx.wait(1)

    weight_hash = decode_utf8(retrieve_aggregated_weights_tx)

    # download the aggregated weights from IPFS
    start_time = time.time()
    aggregated_weights_encoded = IPFS_client.cat(weight_hash)
    print("IPFS 'cat' time: ", str(time.time() - start_time))

    aggregated_weights = weights_decoding(aggregated_weights_encoded)

    # setting the model with the new aggregated weights computed by the Manager
    hospitals[_hospital_name].aggregated_weights = aggregated_weights
    hospitals[_hospital_name].model.set_weights(aggregated_weights)


async def main():
    # continuously listens for the CLOSE event from the Blockchain and promptly handles it
    contract_events.subscribe("CloseState", closeState_alert, delay=0.5)
    print("Subscribed to CLOSE...")

    # await for the START event
    coroutine_start = contract_events.listen("StartState")
    print("COROUTINE: waiting START...\n", coroutine_start)
    await coroutine_start
    print("I waited START")
    print_line("_")
    # continue after reception
    start_event()

    # await for the LEARNING event
    coroutine_learning = contract_events.listen("LearningState")
    print("COROUTINE: waiting LEARNING...\n", coroutine_learning)
    await coroutine_learning
    print("I waited LEARNING")
    print_line("_")

    # start of the Federated Learning loop that will be ended by the CLOSE event
    while True:
        round_loop()
        # await for the Manager to send the aggregated weights
        coroutine_AW = contract_events.listen("AggregatedWeightsReady")
        print("COROUTINE: waiting 'send_aggreagted_weights'...\n", coroutine_AW)
        await coroutine_AW
        print("I waited 'send_aggreagted_weights'")
        print_line("_")
        # continue after reception
        aggregatedWeightsReady_event()


asyncio.run(main())
