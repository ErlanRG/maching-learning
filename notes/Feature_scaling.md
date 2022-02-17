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
