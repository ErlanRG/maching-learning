# Difference between simple linear regression, multiple linear regression and polynomial regression

## Simple linear regression

A simple linear regression predicts the value of one variable Y based on
another variable X.

* X: the independent variable
* Y: the dependent variable

Simple linear regression examine the variation between two variables only.

### Example:
Determine the salary of an employee depending based on the number of years.
As number of year goes up, the expectation of the salary should go up as well

We can use the linear equation to model these kind of regressions:

```
y = mX+b
```

## Multiple linear regression

The multiple linear regression examine the variation between more than two variables.
Can be represented by the following equation:

```
y = b0+m1*x1+m2*x2+...+mn*xn
```

## Polynomial regression

The polynomial regressions models the relationship between the independent
variable X and the dependent variable Y with the following equation:

```
y = b0+b1*x+b2*x^2+...+bn*x^n
```

---

# Importing datasets

* To import a csv file and assign it to a variable:

```
dataset = pd.read_csv()
```

* To determine columns and rows use the `iloc` function:

```
x = dataset.iloc[].values
```

`iloc` receives two parameters. It could be ranges. For this particular case,
the syntax would be:

```
x = dataset.iloc[<rows>, <columns>].values
```

And then it will return the values of each point.

---

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

1. The ***training set*** is applied to train, or fit, your model. For example, you
   use the training set to find the optimal weights, or coefficients, for
   linear regression, logistic regression, or neural networks.

2. The ***validation set*** is used for unbiased model evaluation during
   hyperparameter tuning. For example, when you want to find the optimal number
   of neurons in a neural network or the best kernel for a support vector
   machine, you experiment with different values. For each considered setting
   of hyperparameters, you fit the model with the training set and assess its
   performance with the validation set.

3. The ***test set*** is needed for an unbiased evaluation of the final model. You
   shouldn’t use it for fitting or validation.


Source: [RealPython - Split Your Dataset With scikit-learn's train_test_split()](https://realpython.com/train-test-split-python-data)

---

# Feature scaling

## What is it?
It is a step of Data Pre Processing that is applied to independent variables or
features of data. It basically helps to normalize the data within a particular
range. Sometimes, it also helps in speeding up the calculations in an
algorithm.

### Package used
```
sklearn.processing
```

### Import
```
from sklearn.preprocessing import StandardScaler
```

## Why and Where to Apply Feature Scaling?
The real-world dataset contains features that highly vary in magnitudes, units,
and range. Normalization should be performed when the scale of a feature is
irrelevant or misleading and not should Normalise when the scale is meaningful.

The algorithms which use Euclidean Distance measures are sensitive to
Magnitudes. Here feature scaling helps to weigh all the features equally.

Formally, If a feature in the dataset is big in scale compared to others then
in algorithms where Euclidean distance is measured this big scaled feature
becomes dominating and needs to be normalized. 

### Examples of Algorithms where Feature Scaling matters 
1. K-Means uses the Euclidean distance measure here feature scaling matters. 
2. K-Nearest-Neighbours also require feature scaling. 
3. Principal Component Analysis (PCA): Tries to get the feature with maximum
   variance, here too feature scaling is required. 
4. Gradient Descent: Calculation speed increase as Theta calculation becomes
   faster after feature scaling.

* ***Normalization*** is recommended when you have a normal distribution in most of
  your features (specific situations).
* ***Standarization*** works all the time

Source: [GeeksforGeeks - Python | How and where to apply Feature Scaling?](https://www.geeksforgeeks.org/python-how-and-where-to-apply-feature-scaling)

---
