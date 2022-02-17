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
