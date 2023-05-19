import os
import sys

# Get the directory containing this script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Add the directory to sys.path
sys.path.insert(0, dir_path)

from brownie import FederatedLearning, accounts
from helpful_scripts import get_account
import fl_deploy

from utils_simulation import createHospitals, set_hospitals, get_hospitals

# if it is the first run, write "main" after "scripts/setup.py" in CLI
isCreated = True
if "main" in sys.argv:
    isCreated = False

def main():
    """
    Hospitals creation
    """
    hospitals = None
    if isCreated:
        hospitals = get_hospitals()
    else:
        hospitals = createHospitals()

    """
    KYC Process and Off-Chain Hospitals Registration
    """


    """
    Blockchain implementation
    """
    fl_deploy.deploy_federated_learning()

    """
    Assign blockchain addresses to Hospitals
    """
    # only with fl-local network
    for idx, hospital_name in enumerate(hospitals, start=1):
        hospitals[hospital_name].address = accounts[idx].address
        print(idx, hospitals[hospital_name].address)

    for hospital in hospitals:
        print(hospital)
    
    """
    Opening the Blockchain and Adding the Collaborators
    """
    federated_learning = FederatedLearning[-1]
    manager = get_account()

    # print(federated_learning.get_state())
    open_tx = federated_learning.open({"from": manager})
    open_tx.wait(1)
    print(open_tx.events)
    # print(federated_learning.get_state())
   

    for hospital_name in hospitals:
        hospital_address = hospitals[hospital_name].address
        add_tx = federated_learning.add_collaborator(hospital_address, {"from": manager})
        add_tx.wait(1)
        print(add_tx.events)
    
    if not isCreated:
        # Print all information of each Hospital object in the dictionary
        for key, hospital in hospitals.items():
            print(f"Key: {key}")
            hospital_info = vars(hospital)
            for attr, value in hospital_info.items():
                print(f"{attr}: {value}")
            print()

        set_hospitals(hospitals)

if __name__ == "__main__":
    main()