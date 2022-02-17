# Data splitting

One of the key aspects of supervised machine learning is model evaluation and
validation. When you evaluate the predictive performance of your model, it’s
essential that the process be unbiased. Using train_test_split() from the data
science library scikit-learn, you can split your dataset into subsets that
minimize the potential for bias in your evaluation and validation process.

## The Importance of Data Splitting

Supervised machine learning is about creating models that precisely map the
given inputs (independent variables, or predictors) to the given outputs
(dependent variables, or responses).

What’s most important to understand is that you usually need unbiased
evaluation to properly use these measures, assess the predictive performance of
your model, and validate the model.

This means that you can’t evaluate the predictive performance of a model with
the same data you used for training. You need evaluate the model with fresh
data that has not been seen by the model before. You can accomplish that by
splitting your dataset before you use it.

## Training, Validation, and Test Sets

1. The ***training set*** is applied to train, or fit, your model (based on
   existing observations) For example, you use the training set to find the
   optimal weights, or coefficients, for linear regression, logistic
   regression, or neural networks.

2. The ***validation set*** is used for unbiased model evaluation during
   hyperparameter tuning. For example, when you want to find the optimal number
   of neurons in a neural network or the best kernel for a support vector
   machine, you experiment with different values. For each considered setting
   of hyperparameters, you fit the model with the training set and assess its
   performance with the validation set.

3. The ***test set*** is needed for an unbiased evaluation of the final model.
   You shouldn’t use it for fitting or validation. ***Evaluate the performance of
   the model based on new observations.***

Source: [RealPython - Split Your Dataset With scikit-learn's train_test_split()](https://realpython.com/train-test-split-python-data)
