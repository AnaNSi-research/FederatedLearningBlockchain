# Federated learning on a blockchain of hospital peers
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
This is just a simulation. For concurruncy problems on training on the same gpu, the collaborator.py script contains a loop that trains the
different hospital model instances one at time in sequence. In a real time scenario with more than one peer it is possible to run 
the different learnings at the same time and it works in the same way.
### run collaborator
`brownie run .\scripts\collaborator.py --network fl-local`

### run federated_learning
#### another shell
`brownie run .\scripts\federated_learning.py --network fl-local`
