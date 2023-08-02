# Federated Learning on Blockchain with Hospital Peers for Alzheimer's MRI Image Classification
## Abstract
This project is about the application of **federated learning** techniques on a **blockchain**, for learning a model that can be used by hospitals
for the classification of MRI images of Alzheimer patients. It is based on the ensamble models technique applied in the weight space of neural networks 
(not with different model instances and then just averaging together the scores given as output by the last layer of the neural networks).
This aims at reducing the **variance** and also the **bias** at the same time, increasing the capacity
of the aggregated model to fit the training data and reducing at the same time the difference in performance of the different neural model instances of the 
hospitals due to the different datasets they have.
For this purpose the hospitals do not share and upload the datasets on the blockchain for two reasons:
* security and privacy concernes
* storage capacity issues

Thus, they share just the model weights they obtain from each federated learning round. Then they upload the weights on the blockchain (actually the weights
are stored on IPFS and what is loaded in the blockchain is just the hash of the weights) that will be aggregated together and so on so forth.

This project focuses on the implementation of **federated learning** techniques within a **blockchain** framework to create a collaborative model for classifying MRI images of Alzheimer's patients. The primary objective is to enhance the model's performance by leveraging ensemble models in the weight space of neural networks rather than simply averaging the scores of different model instances.

The main advantages of this approach are twofold:

1. **Reduced Variance and Bias**: By employing ensemble techniques in the weight space, the aggregated model achieves a more balanced trade-off between variance and bias. This leads to improved generalization and better performance on unseen data.

1. **Privacy and Security**: Addressing privacy concerns, hospitals do not share or upload the raw datasets to the blockchain. Instead, they only share the model weights obtained from each federated learning round. This ensures that sensitive patient data remains protected.

Additionally, the approach addresses storage capacity issues as the weights are stored on IPFS (InterPlanetary File System), and only the hash of the weights is loaded onto the blockchain for aggregation.

The key steps of the process are as follows:

1. Hospitals participate in federated learning rounds and train their respective models locally on their datasets.
1. Model weights, not raw data, are shared by the hospitals after each round.
1. The shared model weights are securely uploaded onto the blockchain (with the actual weights stored on IPFS).
1. The blockchain aggregates the weights, leading to the creation of an improved, collaborative model.
1. The process is iterated over multiple rounds to continuously improve the model's performance.

By adopting this federated learning approach on a blockchain, hospitals can collectively benefit from a more powerful and privacy-preserving model without directly sharing sensitive data. This contributes to a better understanding and classification of Alzheimer's disease, even when dealing with diverse datasets from different hospital sources.

## Setup
This setup is just for a simulation
### Requirements
* Ganache
* IPFS
* Miniconda
  * eth-brownie
  * cuda
  * tensorflow
  * opencv-python
  * pandas
  * scikit-learn

### base deactivate
`conda deactivate`

### environment creation
`conda create --name blockchain_project python=3.9`

### activation
`conda activate blockchain_project`

### pip update
`python -m install pip --upgrade pip`

### cuda installation
`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

### tensorflow=2.10 installation
`pip install "tensorflow<2.11"`

### opencv-python installation
`pip install opencv-python`

### pandas installation
`pip install pandas==1.5.3`

### eth-brownie installation
`pip install eth-brownie`

### scikit-learn installation
`pip install scikit-learn`

### ganache installation
https://trufflesuite.com/ganache/

### ipfs installation
https://github.com/ipfs/ipfs-desktop/releases

### add network brownie
`brownie networks add Ethereum fl-local host=http://127.0.0.1:7545 chainid=5777 timeout=3600`

### check network
`brownie networks list`

### setup first time
`brownie run .\scripts\setup.py main --network fl-local` 
#### setup after first time
`brownie run .\scripts\setup.py --network fl-local`

## Running
This is just a simulation. For concurruncy problems on training on the same GPU, the _collaborator.py_ script contains a loop that trains the
different hospital model instances one at time in sequence. In a real time scenario, with more than one peer, it is possible to run 
the different learnings at the same time and it works in the same way.
### run collaborator
`brownie run .\scripts\collaborator.py --network fl-local`

### run federated_learning
#### another shell
`brownie run .\scripts\manager.py --network fl-local`
