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
- Normal vs Uniform distribution -> Uniform aka Rectangular distrbution: has constant probability.
- Random distribution has a numbers generated but in a bell-shaped manner.
- Having a different seed value while perfoming normal distribution gives us different values for the randomly generated tensors.
* seed(42) eliminates some form of randomness while generating text.

## Reproducible Randomness
- Have both a global and operation level seed set.



# Numpy Array vs TF Tensors
- tensors can be run on GPU hence faster

# Shuffling:
This is important given:
    Eliminates biasness: Given 15000 photos of tea 'n chapo
    Say 1000 of chapo and 5000 of tea.
    If the neural net is to learn from the first 1k photos of chapo
    It will have no idea how tea looks like, hence the shuffling allows the model
    learn better, 200- chapos, 450- tea etc.


> ### X_TRAIN : UPPERCASE == ARRAY
> ### y_train : lowercase == vector

# Tensor Attributes
* Shape : length/no. of elements of each of the dimensions of a Tensor
* Size : total number of items in the tensor
* Rank : no. of tensor dimensions. 'tf.ndim'
* Dimension/Axis : Access a particular dimension of a tensor

> Most of the Time: We spend in shaping our tensors for either entry into a model or understanding the outputs of a model, so that we can shape our inputs.


## Rules for Matrix/Tensor Muliplication
> `1. The inner dimensions must match [3,2] and [2,3] : 2 & 2 are the inner dims `

> `2. The resulting matrix has the shape of the outter dimension`
* By transposing (X) or Y, the outter dimensions are different, hence an observable change in both the outresulting matrix size.

## Why tf.reshape and tf.transpose is different.
> `transpose flips the axis, while reshape, shuffles the tensors into the shape I want`

# The Dot Product.
`tf.tensordot()`

### So when do we do transpose or reshape
> 1. Its done in the background for us.

> 2. Most of the time, operation on a tensor is `Transposition`

# Changing the dtype of a tensor
> `default dtype either float/int32`

# Aggregation
> `gather data and express in summary form`
* Minimum
* Maximum
* Mean
* Sum

### `Standard Deviation`
>   `measure of dispersion for a dataset`

>   `the squareroot of variance`