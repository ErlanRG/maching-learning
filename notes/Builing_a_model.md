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
