# Goal
In this series of practices, we will go through the theory and concepts of machine learning along with the applications by using Python.

## Algorithm type
Below are the different types of the machine learning algorithms:

#### Prediction
  - Supervised
    - Continuous variables: Regression
      - Gradient Descent
      - Gradient Ascent
    - Discrete variables: Classification
      - kNN
      - Decision Tree
      - Random Forest
#### Explore
  - Unsupervised
     - Fit data into discrete groups: Clustering
       - Agglomerative
       - Divisive

# Regression
We will go through the theory and applications of Gradient Descent and Gradient Ascent.

## Gradient Descent
First, we start with the Gradient Descent. When doing the gradient descent, we basically keep updating the parameter and adjusting the function to find the model with the lowest cost/ error to solve the problem of regresiion. Now, we are going to use the kangaroo’s nasal dimension data to estimate the optimal intercept and gradient for this prediction.

### Variables 
```
X: nasal length of the Kangaroos in mm
Y: nasal width of the Kangaroos in mm
```

### 0. Set up
Load library and import file.
```
import numpy as np
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
df = pd.read_csv('kangaroo.csv')
```
### 1. Variable setting
Define the variables and the outcome you want to calculate

```
X = df[['X']]
y = df[['Y']]
```
### 2. Run Gradient Descent
Create gradient descent function
```ruby
def grad_descent(X, y, alpha, epsilon):
  iteration = [0]
  i = 0
  m = np.shape(X)[0] # Total number of samples

  # Initialize the parameters
  Theta = np.ones(shape=(len(df.columns),1))

  # Compute the initial cost
  cost = [np.transpose(X@Theta-y)@(X@Theta-y)]
  delta = 1

  while(delta>epsilon):
    gradient = ((np.transpose(X))@(X@Theta-y))
    Theta = Theta - alpha*gradient
    J_Theta = 1/m*(np.transpose(X@Theta-y)@(X@Theta-y))
    print(J_Theta[0])
    cost.append(J_Theta)
    delta = abs(cost[i+1]-cost[i])
    if ((cost[i+1]-cost[i])>0):
      print("The cost is increasing. Try reducing alpha.")
      break
    iteration.append(i)
    i += 1
  print("Completed in %d iterations." %(i))
  return Theta, cost
```
Concat dataframe and convert into array
```ruby
X = pd.concat((pd.DataFrame([1,2]*23),df[['X']]),axis=1, join='inner').to_numpy()
y = y.to_numpy()
```
Run gradient descent. Set and adjust the alpha and epsilon.
```ruby
Theta = grad_descent(X=preprocessing.scale(X), y=y, alpha=0.01, epsilon=10**-10)
```
Print gradient descent plot which shows the change of the cost on each iteration
```ruby
cost_vals = [val[0,0] for val in cost]
plt.plot(range(2, len(cost_vals)+1),cost_vals[1:], marker='o')
plt.xlabel("Number of Iterations (Epochs)")
plt.ylabel("Cost function J(theta)")
plt.title("Gradient Descent")
plt.show()
```
<img width="621" alt="Screenshot 1402-11-17 at 16 41 22" src="https://github.com/kellyhuanyu/Machine-Learning-Theory/assets/105426157/c2fdc89d-8f06-4877-a015-5c91f1758a5c">

## Gradient Ascent



















