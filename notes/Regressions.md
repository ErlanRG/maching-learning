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
