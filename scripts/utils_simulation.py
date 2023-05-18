import numpy as np
import sys
import os
import cv2

HOSPITALS_FILE = "/home/imbo/Unibo/Blockchain/fl_project/off_chain/hospitals.pkl"
DATASET_PATH = "/home/imbo/Unibo/Blockchain/Database/Alzheimer_s Dataset Classes/"
DATASET_LIMIT = None
TEST_VAL_SPLIT = 0.30
RANDOM_SEED = 42
HOSPITAL_SPLIT = {"Alpha": 0.5, "Beta": 0.3, "Gamma": 0.2}
# HOSPITAL_SPLIT = {"Alpha": 0.33, "Beta": 0.33, "Gamma": 0.34}

import numpy as np
import pandas as pd
import pickle
from classHospital import Hospital
from sklearn.model_selection import train_test_split


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    if DATASET_LIMIT:
        for dir1 in os.listdir(img_folder):
            for idx, file in enumerate(os.listdir(os.path.join(img_folder, dir1))):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, 0)
                image = np.array(image)
                image = image.astype("float32")
                img_data_array.append(image)
                class_name.append(dir1)

                if idx == DATASET_LIMIT:
                    break
    else:
        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, 0)
                image = np.array(image)
                image = image.astype("float32")
                img_data_array.append(image)
                class_name.append(dir1)
    return img_data_array, class_name

def createHospitals():
    hospitals = {}

    # extract the image array and class name
    img_data, class_name = create_dataset(DATASET_PATH)

    # target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    target_dict = {
        "NonDemented": 0,
        "VeryMildDemented": 1,
        "MildDemented": 2,
        "ModerateDemented": 3,
    }

    target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

    X = np.array(img_data, np.float32)
    y = np.array(list(map(int, target_val)), np.float32)

    rows = len(X)
    values_list = []
    for hospital_name in HOSPITAL_SPLIT:
        values_list += [hospital_name] * int(rows * HOSPITAL_SPLIT[hospital_name])

    print(type(X))
    print(type(X[0]))
    print(X.shape)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]


    print(type(X))
    print(type(X[0]))
    print(X.shape)

    df = pd.DataFrame({"X": list(X), "y": list(y)})
    if df.shape[0] != len(values_list):
        values_list.append("Gamma")
    print(values_list)
    df["hospital"] = values_list
    # df['hospital'] = df['hospital'].map(hospitals)

    for hospital_name in HOSPITAL_SPLIT:
        print(hospital_name)
    print(HOSPITAL_SPLIT.keys())
    print(list(HOSPITAL_SPLIT.keys()))

    dataset = dict.fromkeys(list(HOSPITAL_SPLIT.keys()))
    print(dataset)

    for hospital_name in HOSPITAL_SPLIT:
        X_h = df[df["hospital"] == hospital_name]["X"].to_numpy()
        y_h = df[df["hospital"] == hospital_name]["y"].to_numpy()

        X_h = np.stack(X_h, axis=0)
        y_h = np.stack(y_h, axis=0)

        dataset[hospital_name] = {}

        (
            dataset[hospital_name]["X_train"],
            X_test,
            dataset[hospital_name]["y_train"],
            y_test,
        ) = train_test_split(X_h, y_h, test_size=TEST_VAL_SPLIT, random_state=RANDOM_SEED)
        (
            dataset[hospital_name]["X_test"],
            dataset[hospital_name]["X_val"],
            dataset[hospital_name]["y_test"],
            dataset[hospital_name]["y_val"],
        ) = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED)

        hospitals[hospital_name] = Hospital(
            hospital_name, dataset[hospital_name]
        )

    """
    hospital_Alpha = Hospital("Alpha", dataset_Alpha)
    hospital_Beta = Hospital("Beta", dataset_Beta)
    hospital_Gamma = Hospital("Gamma", dataset_Gamma)
    """

    print(X_test.shape)
    print(y_test.shape)

    # test

    for h in hospitals:
        for n in hospitals[h].dataset:
            print(
                h,
                n,
                hospitals[h].dataset[n].shape,
                type(hospitals[h].dataset[n]),
                hospitals[h].dataset[n][0].shape,
                type(hospitals[h].dataset[n][0]),
            )

    return hospitals

def get_hospitals():
    hospitals = {}
    with open(HOSPITALS_FILE, "rb") as file:
        hospitals = pickle.load(file)
    return hospitals

def set_hospitals(hospitals):
    serialized_hospitals = pickle.dumps(hospitals)
    with open(HOSPITALS_FILE, "wb") as file:
        file.write(serialized_hospitals)

def get_X_test():
    hospitals = get_hospitals()
    X_test = None
    try:
        X_test = np.concatenate(
            [hospital.dataset["X_test"] for hospital in hospitals], axis=0
        )
    except ValueError as e:
        print("ERROR --> catch at:", __name__, "on file:", __file__, "\nException:", str(e))
    if not X_test:
        raise Exception("ERROR --> catch at:", __name__, "on file:", __file__, "\nException: X_test is empty")
    return X_test
        
def get_y_test():
    hospitals = get_hospitals()
    y_test = None
    try:
        y_test = np.concatenate(
            [hospital.dataset["y_test"] for hospital in hospitals], axis=0
        )
    except ValueError as e:
        print("ERROR --> catch at:", __name__, "on file:", __file__, "\nException:", str(e))
    if not y_test:
        raise Exception("ERROR --> catch at:", __name__, "on file:", __file__, "\nException: y_test is empty")
    return y_test



def print_weights(weights):
    print(len(weights))
    print(type(weights))
    for w in weights:
        print(w.shape)
        print(type(w))
        print(str(sys.getsizeof(w)))
    print("weights size:" + str(sys.getsizeof(weights)))
    print(
        "weights TOTAL size:"
        + str(sys.getsizeof(weights) + sum(sys.getsizeof(w) for w in weights))
    )
    print("---------------------------------------------")

def print_listed_weights(weights_listed):
    print(len(weights_listed))
    print(type(weights_listed))
    for w in weights_listed:
        print(len(w))
        print(type(w))
        print(str(sys.getsizeof(w)))
    print("weights_listed size:" + str(sys.getsizeof(weights_listed)))
    print(
        "weights_listed TOTAL size:"
        + str(
            sys.getsizeof(weights_listed)
            + sum(sys.getsizeof(w) for w in weights_listed)
        )
    )

