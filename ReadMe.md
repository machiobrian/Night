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

To pass into a Neural Net, we have to
* Encode (*turn input (image, sound, text) into numbers*)
* Numerical Data - Tensor
* Feed tensor into NN.
* Neural net learns the ~Representation: *patterns, features, weights*
* Creates some Representation outputs. - Tensors still
* Turn the tensors into human readable form.

## Inputs -> Numerical Encoding -> Learns Representation (NN) -> Representation O/p -> Outputs

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
- Scalar : a single number; have no dimension
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
* Shape : length/no. of elements of each of the dimensions of a Tensor `[Also, the number of Input vectors]`
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

### Positional Maximum and Minimum.
* It is as basic as, at what point/`[index]`/position, in the tensor does the minimum and maximum value occur

## Squeezing a Tensor
* Removes dimensions of size 1 from the shape of a tensor.

### One-Hot Encoding
*this is a form on Numerical encoding*
> `Setting every other value to 0 in a row, and just one value to 1`

>          Green Blue Red
>            1   0   0 
>            0   1   0 
>            0   0   1

### Always check the datatypes:
> `tf.constant() : This method of creating an array has the default as float/int 32`

> `np.array([]): This method has the default as float/int 64`

### Regression Problem
> `Answer the questions "How much/many" "numbers - coordinates of (bounding boxes)"`
* Also, dependent/`output` (outcome vars) and independent/`input` (predictors, covariates, features) variables `relationship`

> #### `Always Note: Most of the Machine Learning task is in Tweaking the Inputs to be fed in a Model and Observing/Investigating and Studying the model's Output` - Defining Inputs and Outputs

# Hyperparameters
> A parameter set prior to training a model.
* Their values cannot be estimated from data - trial and error.
* Techniques trial and error: `Grid Search & Random Search.`
* Techniques hyperparam optimization: `Bayesian optimization`
> Hyperparams: `learning rate, no. of hidden-units(layers and neurons), activition function, optimizer {how well the model is to learn and improve the loss function}, loss function {tells our model how wrong we are}`

# Model Improvements:
Can be done while
> Creating a Model
* change activation function of each layer
* increase the number of units (hidden layers and neurons) 
> Compiling a Model
* change the optimization function
* change the learning-rate of an optimization function
> Fitting a model
* have more epochs `increase the number of times the model goes through the training data`

>> Have Training datasets in percentages of say 10

* This allows one to first run a model, `a number of times` on smaller datasets, taking shorter times, hence making observations on performance before training it on large dataset
* Allows running as many experiments as possible, to figure out what dosesn't work, before increasing our params

* The `Optimizer_learning_rate` is by far most the most important hyperparam to tune
* The Simplest to tune is the `number of epochs`
* The number of `hidden_layers`, the `activation_fxn` are also important.

`Evaluating a Model` : visualize visualize visualize
* The data - what data are working with
* The Model - what does it look like
* The Training - how it perfomes as it learns
* The Predictions - how they lines up with ground truth (labeled data)

`Sets`
* Training Set - the model is taught on this piece of data
* `Validation Set - the model is tweaked/tuned on this data, typically 10-15% of available data`
* Test Set - this is where the model is evaluated, we test what it has learnt


`Visualizing a Model before it is fit()`
* Always specify the `input_shape` on the first layer.
* Create, Compile the Model as usual
> `Parameters`: These are the patterns that the model is going to learn, also, the total number of parameters in the model

> `Hyper-Parameters` : The tweakable/alterable/adjustable values before training a model

### After Visualizing a Model: Definitions

* Total Params - total number of parameters in the model. Also, the patterns the model is supposed to learn from.
* Trainable params - the parameters or the patterns the model can updaete as it runs.
> While `importing an already trained model`, we `freeze` the `learnt patterns`, therefore we may have both trainable and non-trainable params.
* Non-Trainable params - already learnt patterns that we do not want to update them during training; the ones we freeze `Non-trainable params makes more sense during Transfer Learning.`

> `An increase in number of hidden layers increases the number of non/trainable params, also, increasing the number of shapes for a single [Dense] layer`
* Weight Matrix and a Bias-vector

### Visualizing our Model's Predictions
* Its a great idea to plot the `predictions` against `ground_truth`


### Regression Tasks
`MAE` - Mean Absolute Error: 'on average, how wrong is each of my models predictions'
`MSE` - Mean Square Error: 'square the average errors' - when having large erros
`Huber` - basically a combo of MSE and MAE. Less sensitive to outliers.

> |labels - predictions|

* Squeezing and Adding a dimension are both ways to make models fit either for evaluations or any other task:

`Quiz:`
	while evaluating a model performance, when do we squeeze and when do we expand dimensions ?
	
`Ans:`
	it all depends on the models structure.


## `On the First Dense input layer, always learn to define the input_shape=[1]`

> Note: Building a model is all about experimenting and experimenting, therefore, learn to start with small model then make sure they work before proceeding to large models. This minimizes the waiting time between model training


# Tracking our Experiments
* Allows monitoring and tracking the performance of our experiments
> TensorBoard - built into tf, 

> Weights and Biases - a tool for tracking all kinds of ML experiments: plugs into tensorboard

# Saving a Model  
`Saved-Model Format` - saved using `model.save(file_name)` can be loaded using tf.keras.model.load_model
`HDF5 format` - saved using `model.save(file_name)` 
> * the hdf5 is a universal format, allows use in other applications - heirachial data format

### Saving a file allows us to:
* Save the model's arhitecture, allowing to re-instantiate the model
* Save the model weights: the patterns the model learnt
* The State of the Optimizer - allows us to resume the training where we left-off

#------------------------------------------------------------------------------------------------------------------------------------#
### Medical Cost Estimation:
* We are using features/predictors to estimate the cost of medical charges
* Since we are dealing with numbers its a Linear Regression Problem.
* age sex bmi children smoker region - `inputs,features,predictors` - independent variables
* charges - `the outcome we are predicting` - dependent variables

`sklearn's test_train_split` divides our dataset (X,y) into 4 respectively:
* X_train
* X_test
* y_train
* y_test
> Always note: All manner of training comes before testing.

* Hyperparameters aka Optimizer: Tuning often involves choosing the right optimizer/optimizer values
* `Learning-rate` - Defines the adjustment of our weights wrt loss gradient descent. `Determines how fast/slow we move towards optimal weights` "How fast does it take you to adjust to new content, also how unbiased are you.
> Low lr - I have known dogs to be black therefore, all dogs are black. A white one, ain't really a dog but an outlier

> High lr - I have known dogs to be black, now that I have seen a white one, all dogs are supposed to be white, forget about black.

> Desirable lr - when shown a white dog, he would quickly understand that black is not an important feature of dogs and would look for another feature.

## Other forms of Tuning our hyperparamweters
* Normalizing is a thing that can be done on the data to be passed into the model to improve perfomance
#### Normalization aka Scale
* change values of numeric column to a common scale
* does not distort differences in range of values
* most prefered for machine learning
* use MinMaxScaler scikit-learn - set everything btn 1 and 0

#### Standardization
* 
* 

#### Batch_size - normally 32 for tf
* allows us not to run into "out of memory issues"
* images - [batch_size, width, height, color_channels] - tensor representation
# Classification
* Classify one thing as one thing or another
> Types: Binary, Multiclass, Multilabel
* multilabel - each sample has one label
* multiclass - 
* Binary - 1/0 labels
* A classification problem can be: Binary (is it a dog ?)or Multiclass (is this a dog, plate or fruit)
### Loss Function for Classification Problems
* Cross-entropy : common loss function for classification problems
> Binary classification: Binary, Multiclass classification: CategoricalCrossEntropy
