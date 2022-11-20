## Deep Learning in Practice with Python and LUA (VITMAV45) - project work

## Topic

Convolutional Neural Network (CNN) vs Vision Transformer (ViT) for Cloud Image Classification

### Team details

name: Obscure Intelligence

* Barancsuk Lilla - BMMMRS
* Kássa Kristóf - HGNB1P

## Abstract

Convolution Neural Network (CNN) algorithms have been prominent models for image classification, but in recetn years Transformer based methods have also started to gain popularity and usage. In an attempt to get a clear view and understanding of the two architecture for image classification tasks on a cloud dataset of approximately 2000 data points, the project is designed to compare the charasteristics of CNN and Vision Transformer (ViT). For each models we provide a clear and comprehensive review of architectural and functional differences. Then we compare their computing capacity requirements, validation accuracy and training time on an online image dataset with our own implementation of input pipeline from scratch.

## Documentation
main.py
* Cloud image dataset download  and preprocess script.

definitions.py
* Basic category-definitions

load_data.py
* Download script

visualization.py
* Visualization of the data

preprocessing.py
* Data preparation for learning to create teaching, validation and test inputs and outputs.

scaler.py
* Dataset standardization.



## Milestone 1
Open Milestone 1 in Google Colab [here](https://colab.research.google.com/drive/1vHV0-Xz2UidxxIGeMmjiosJxLtjpEcYt?usp=sharing).


## Milestone 2
### Transfomer achitecture
Open the transformer code in Google Colab [here](https://colab.research.google.com/drive/1Gv5sAK2P29KVJ2PEPVUgDEtFI54jpiKj?usp=sharing).

For file preprocessing, run cells under the section "1.) Data preprocessing" in the notebook.

For training the network, run cells under the section "2.) Training ViT model" in the notebook.

For model evaluation, run cells under the section "3.) Evaluating ViT model" in the notebook.

### CNN architecture
Open the CNN code in the same Google Colab [here](https://colab.research.google.com/drive/1Gv5sAK2P29KVJ2PEPVUgDEtFI54jpiKj?usp=sharing).

For file preprocessing, run cells under the section "1.) Data preprocessing" in the notebook.

For training the cnn network, run cells the section "4.) Training CNN model".

For evaluating the cnn network, run cells the section "5.) Evaluating CNN model".
