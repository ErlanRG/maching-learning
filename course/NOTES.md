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

### Assumptions of a linear regression

Before building a linear regression model, we need to check if the following
assumtions are true:

- Linearity
- Homoscedasticity
- Multivariate normality
- Independence of errors
- Lack of multicollinearity

## Multiple linear regression

The multiple linear regression examine the variation between more than two
variables. Can be represented by the following equation:

```
y = b0+m1*x1+m2*x2+...+mn*xn
```

## Polynomial regression

The polynomial regressions models the relationship between the independent
variable X and the dependent variable Y with the following equation:

```
y = b0+b1*x+b2*x^2+...+bn*x^n
```

# Dummy variables

Sometimes, we will face information that does not fit the equation of any of
the above regressions. For example, the categorical data is does not fit in the
equation as variable, but we still need to deal with it. For this kind of
situations we create ***dummy variables***. (See Dealing with categorical data)

## Dummy variable trap

Not all the dummy variables should be included in the model. You can't have the
constant 'b' and all the dummy variables in the same equation. When building
the model, ***ALWAYS OMIT ONE DUMMY VARIABLE***. This is independent of how
many dummy variables you have.

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

# Data processing template analysis

To understand a little bit better the workflow when data processing, these are
the main functions you should use to get to the training and test sets:

### Importing the dataset

1. First, we need to read the file in which the data is contained. The package
  "Panda" has a method to read .csv files. Create a variable to contain the
  data with `panda.read_csv()` method.

```
# Importing datasets
dataset = pd.read_csv('<YOUR_FILE_PATH>')
```

**NOTE**: It is strongly recommended to arrange the raw data in certain way so
it is easy to identify the columns containing the independent variable and the
dependent variable. Rule of thumb is to place the dependent variable in last
column.

2. Second, we have to define variables for the dependent and the independent
   variables. The `.iloc[<ROWS>, <COLUMNS>]` method, from the pandas dataframe,
   will take the indexes of the rows and columns we want to extract from the
   dataset. iloc stands for locate indexes.

```
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

**REMEMBER**: Python can handle ranges. The ":" specifies to take all the
range.
- ":-1" means to take the complete row except for the last one
- "-1" means to take only the last column

### Dealing with missing values

3. Sometimes, the data we are working with has some missing variables. To avoid
   any kind of issues and misscalculations in the future, we need to deal with
   this. One method to deal with the missing data is to replace these missing
   fields with the average of the column in which it is contained.

```
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(<MISSING_VALUES>, <STRATEGY>)
```

We use a library called **scikit-learn** and the class to use is
`SimpleImputer` contained in the `imputerr` method. The output will be a new
matrix of features containing new data, replaceing the missing spots with the
mean or any other strategy chosed The new matrix will be assigned to a new
variable. The`SimpleImputer` method will have two arguments: 
* the missing values (numpy will automatically transform the missing values
  into a "nan" value)
* the strategy you want to choose. For this particular case, we use the "mean"
  of the complete column to replace the empty fields

Up to this point, we haven't connected anything yet; we just've only created an
object with the new values. To first change the new values into the matrix as an
ouput, we use the `fit()` function, and since we only have changed the values
to the independent variable, we will apply the changes to X.

**NOTE**: `fit()` expects all the columns with numerical values **only**, so
that is why the range is from 1:3

```
imputer.fit(x[:, 1:3])
```

Now, to apply the change we just have to replace the range with the new values.
To do this, we use the `transform(<RANGE_TO_TRANSFORM>)` method from the
sklearn package.

```
x[:, 1:3] = imputer.transform(x[:, 1:3])
```

### Dealing with categorical data (dummy variables)

4. If we have categorical data in our dataset, we need to encode this
   information. By default, python would not understand any array of characters
   or strings, but will understand numbers, so this process will take the
   categorical column and will transform it into binary vectors bits. For
   example:

We have three different countries in our categorical column: Spain, France,
Italy. To encode these into three different "columns" (more like binary
vectors), we use the `ColumnTransformer` class from the compose module and the
`OneHotEncoder` class from the preprocessing module. 

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
```

Then we create an object of the new `ColumnTransformer` class. We need to: 

- First, specify what kind of transformation do we want (encoding)
- Second, what kind of transformation we want to do (`OneHotEncoder()`)
- Third, the indexes of the columns we want to transform.

For all the other columns we don't want to take in consideration, we use
`reminder ='passthrough'`.


```
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] ,
                       remainder='passthrough') 
```

To transform and apply the canges all at the same time, we use the method
`fit_transform()` from the `ColumnTransformer` class. We transform the array of
X.


x = np.array(ct.fit_transform(x))

This will create a position for every single category:
- Spain  =  [1, 0, 0]
- France =  [0, 1, 0]
- Italy  =  [0, 0, 1]

5. We also have to deal with the dependent variable in case there is a lable
   there. The process is pretty similar to the previous section, but kind of
   easier:

   This time, we use the `LabelEncoder` class and apply it to the Y variable

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

### Splitting the dataset into training set and test set

6. The `train_test_split()` class from the `model_selection` module allows to
   split the dataset into training set and test set. This will create a pair of
   matrix of features and dependent variable for the trainig set, and another
   pair for the test set.

```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
```

The `train_test_split()` expects the following arguments:

- The combination of matrix of features X and Y. 
- The split size. We are not splitting the dataset into a trainig and test sets
  of the same size. It is recommended to have 80% observations in the training
  set and 20% in the test set.
- There will be random factors during the split, so to have the same amount of
  factors between the trainig set and test set we pass a `random_state`.

---

# Statistical significance

Statistical significance refers to the claim that a result from data generated
by testing or experimentation is not likely to occur randomly or by chance but
is instead likely to be attributable to a specific cause.

[Statistical significance](https://www.investopedia.com/terms/s/statistical-significance.asp) 

## Understanding the P-value and more statistical concepts

- We use the ***p-value*** to determine whether we should reject o fail to
  reject the null hypothesis.
- The ***alpha value (α)*** is the probability of rejecting a null hypothesis
  that is true.
- The ***confidence level*** is how sure we are the confidence interval
  contains the true the population parameter value. This confidence level
  equals 1-α.

---

# Building a model (step by step)

## 5 methods of building models

1. **All-in:** Take in consideration all the variables from a given dataset.
    * Prior knowledge that all the variables will help to predict something
    * You have to use this method
    * Preparing for backward elimination

2. **Backward elimination:**
    1. Select a significance level to stay in the model (e.g. SL = 0.05)
    2. Fit the full mode with all possible predictors
    3. Consider the predictor with the **highest** P-value. If P > SL,
    go to STEP 4, otherwise go to FIN.
    4. Remove the predictor
    5. Fit the model without this variable. Go back to step 3 and
    repeat until you got to FIN and the model is ready.

3. **Forward selection:**
    1. Select a significance leve to stay in the model (e.g. SL =
    0.05).
    2. Fit all simple regression modes **y ~ xn**. Select the one with
    the lowest P-value.
    3. Keep this variable and fit all the possible models with one
    extra predictor added to the one(s) you already have.
    4. Consider the predictor with the **lowest** P-value. If P < SL,
    go to STEP 3, otherwise go to FIN. Go back to step 3 and repeat until you
    got to FIN (FIN: keep the previous model)

4. **Bidirection elimination**:
    1. Select a significance level to enter and to stay in the model.
    E.g: SLENTER = 0.05, SLSTAY = 0.05
    2. Perform the next step forward selection (new variables mus have:
    P < SLENTER to enter)
    3. Perform ALL the steps of Backward elimination (old variables
    must have P < SLTAY to stay). Go to step 2 and repeat. FIN: model is ready.

5. **All possible models**:
    1. Select a criterion of goodness of fit
    2. Construct all possible regression modes: 2^n-1 total combinations
    3. Select the one with the best criterion. FIN: your model is ready

---


