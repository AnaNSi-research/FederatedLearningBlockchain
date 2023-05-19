import os
import sys

# Get the directory containing this script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Add the directory to sys.path
sys.path.insert(0, dir_path)

from utils_simulation import get_X_test, get_y_test, get_hospitals
from utils_manager import create_model,compile_info, HEIGHT, WIDTH, DEPTH, NUM_CLASSES, NUM_EPOCHS

def main():
    X_test = get_X_test
    y_test = get_y_test

    hospitals = get_hospitals()
    X_train = hospitals["Alpha"].dataset["X_train"]
    y_train = hospitals["Alpha"].dataset["y_train"]
    X_val = hospitals["Alpha"].dataset["X_val"]
    y_val = hospitals["Alpha"].dataset["y_val"]


    model = create_model((HEIGHT, WIDTH, DEPTH), NUM_CLASSES)
    model.compile(**compile_info)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=NUM_EPOCHS)

if __name__ == "__main__":
    main()

