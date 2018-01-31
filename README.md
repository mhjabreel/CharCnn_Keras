# Character Level CNN in Keras

This repository contains a Keras implementation for Character-level Convolutional Neural Networks for Text Classification. It can be used to reproduce the results on AG's News Topic Classification Dataset in the following article:
> Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems 28 (NIPS 2015)

## Installation

Install dependencies:

```
$ pip install -r requirements.txt
```

## How to use

1. Specify the training and testing data sources and model hyperparameters in the config.py file.

2. Run the main.py file as below:

```sh
$ python main.py
```
