# symmetrical-knowledge-distillation-in-Auto-encoder-networks
Knowledge distillation framework by linking symmetrical components of auto-encoders to improve bidirectional domain transfer

# Knowledge Distillation

The folder contains codes for self-distillationfor classification of the CIFAR-10 dataset and self distillation on an auto-encoder. Details on the architectures are mentioned in the report. 


## Preparing the CityScapes Dataset

First Download the CityScapes Dataset from https://www.cityscapes-dataset.com/

Separate the two images in the train and val folders and put them in different folders using the code split.py. Change the directory names in split.py accordingly. 

Change the names of the test directory in all the below codes also. 


1) Trial1.py: Self distillation between encoder 1 and decoder 4 through maxpool

## Training

python trial1.py --mode train

## Testing
 
python trial1.py --mode test


2) Trial2.py: Self distillation between encoder 1 and decoder 4  and encoder 2 and decoder 3 through maxpool

## Training

python trial2.py --mode train

## Testing
 
python trial2.py --mode test


3) Trial1.py: Self distillation between encoder 1 and decoder 4 through convolution

## Training

python trial3.py --mode train

## Testing
 
python trial3.py --mode test

