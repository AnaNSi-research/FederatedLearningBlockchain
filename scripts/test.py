import numpy as np

def main():
    foo = {}

    try:
        bar = np.concatenate(
            [fo.dataset["X_test"] for fo in foo], axis=0
            )
    except ValueError as e:
        print("ERROR: ", str(e))