# Deep Learning
- Turn Data to Numbers and Find deep patterns.
- Parameters also known as features.

ML - Works best w/structured data
DL - works best w/unstructred data

# Common Algos for ML:
- Random Forest
- Naive Bayes
- SVM
- Nearest neighbour

# Common Algos for DL:
- Neural Nets
- Fully connected Neural net/ Dense
- Conv Neural net
- Recurrent nueral net
- Transformers - (Architecture)

*Depending on how the data is prepped, we can use any Algo.*

# Neural Nets
Network of neurons 

To pass into a Neural Net, we have to encode (*turn input (image, sound, text) into numbers*)
Numerical Data - Tensor
Feed tensor into NN.
Neural net learns the ~Representation: *patterns, features, weights*
Creates some Representation outputs. - Tensors still
Turn the tensors into human readable form.

# Inputs -> Numerical Encoding -> Learns Representation (NN) -> Representatio O/p -> Outputs

Input Layer -> Hidden Layer -> Output Layer. 

# Patterns
Can be used Interchngibly with:
- Embedding
- Weights
- Feature representation
- Feature vectors

# Types of learning:
Semi/Supervised : some/labels present, respectively
Unsupervised : No label
Transfer : Patterns can be transfered from one dataset.

# seq2seq problems:
- Translation : sequence of words
- Speech recognition - sequence of waves

# classification/regression problems:
is it 1 or 0 ?

# TensorFlow
- ML platform
- Always check TF Hub (*leverage transfer learning*).


# Tensor 
- Numerical way to rep info (*input/output*)

- Fitting a Model to Data : Learning Patterns
- Preprocessind data : Getting data into tensors

# TensorFlow WorkFlow:
- Get Data ready: turn it into a tensor
- Build or Pick a Pretrained Model
- Fit the model to the Data & Leanr to make predictions
- Evaluate the Model's performance
- Improve the model through experimantation
- Solving and Loading the Model.


# Definitions:
- Scalar : a single number
- Vector : a number w/direction
- Matrix : a 2-D array of numbers
- Tensor : an n-dim array of numbers

# Tensors 
There are 
- variable tensors
- constant tensors 
what to use and where is decided by tensorflow


# Random Tensors
- Having the seed present (st seed for reproducibility.)
- To have reproducible experiences, it is important to have the shuffling done in the same order no matter the number of times, we seek to run our data through the net.
This is made possible by setting the random seed at a both the Global and Operational level.


# Numpy Array vs TF Tensors
- tensors can be run on GPU hence faster
