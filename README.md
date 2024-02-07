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
#### Exploration
  - Unsupervised
     - Fit data into discrete groups: Clustering
       - Agglomerative
       - Divisive

# Regression
We will go through the theory and applications of Gradient Descent and Gradient Ascent.

## Gradient Descent
First, we start with the Gradient Descent. When doing the gradient descent, we basically keep updating the parameter and adjusting the function to find the model with the lowest cost/ error to solve the problem of regression. Now, we are going to use the kangaroo’s nasal dimension data to estimate the optimal intercept and gradient for this prediction.

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
While the outcome from the gradient descent will be linear, continuous number, what about the outcome that is categorical? We will then need logistic regression to find the model.
For example, if you have a weather dataset with the following attributes: humidity, temperature, and wind speed, each is describing one aspect of the weather for a day. And based on these attributes, you want to predict if the weather for the day is suitable for playing golf. In this case, the outcome variable that you want to predict is categorical (here, ‘yes’ or ‘no’). 

We can create a logistic regression model to predict the results by analyzing an automated answer-rating site marks each post in a community forum website as “good” or “bad” based on the quality of the post. 

### Variables
```
i.     num_words: number of words in the post
ii.    num_characters: number of characters in the post
iii.   num_misspelled: number of misspelled words
iv.    bin_end_qmark: if the post ends with a question mark
v.     num_interrogative: number of interrogative words in the post
vi.    bin_start_small: if the answer starts with a lowercase letter. (‘1’ means yes, otherwise no)
vii.   num_sentences: number of sentences per post
viii.  num_punctuations: number of punctuation symbols in the post
ix.    label: the label of the post (‘G’ for good and ‘B’ for bad) as determined by the tool.
```

### 0. Set up
Load library and import file.
```
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
```
```
df = pd.read_csv('quality.csv')
```
### 1. Variable setting
Define the variables and the outcome you want to calculate

```ruby
X = df[['num_words', 'num_characters', 'num_misspelled', 'bin_end_qmark', 'num_interrogative', 'bin_start_small',
      'num_sentences', 'num_punctuations']]
df['label'] = df['label'].replace({'B': 0, 'G': 1})
y = df[['label']]
```
### 2. Run Gradient Ascent
Concat dataframe and convert into array
```ruby
X = pd.concat((pd.DataFrame([1,2]*28),df[['num_words', 'num_characters', 'num_misspelled', 'bin_end_qmark', 'num_interrogative', 'bin_start_small',
      'num_sentences', 'num_punctuations']]),axis=1, join='inner').to_numpy()
y = y.to_numpy()
```
Split the dataset into training & testing dataset
```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```
To maximize the goodness of the model, we need to maximize the likelihood by logging the likelihood of the model to best explain the data.
```ruby
logmodel = sm.Logit(y_train, X_train).fit(disp=True)
print(logmodel.summary())
```
Result:
```
Optimization terminated successfully.
         Current function value: 0.352070
         Iterations 9
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                   19
Model:                          Logit   Df Residuals:                       10
Method:                           MLE   Df Model:                            8
Date:                Fri, 19 Jan 2024   Pseudo R-squ.:                  0.4827
Time:                        07:11:44   Log-Likelihood:                -6.6893
converged:                       True   LL-Null:                       -12.932
Covariance Type:            nonrobust   LLR p-value:                    0.1308
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
x1            -0.0960      0.938     -0.102      0.919      -1.934       1.742
x2             0.2094      0.173      1.211      0.226      -0.129       0.548
x3             0.0008      0.042      0.019      0.985      -0.082       0.083
x4            -1.0360      0.714     -1.451      0.147      -2.435       0.363
x5            -3.1917      3.238     -0.986      0.324      -9.537       3.154
x6            -0.1684      0.301     -0.560      0.575      -0.757       0.421
x7             0.6691      2.254      0.297      0.767      -3.749       5.087
x8            -0.7372      1.994     -0.370      0.712      -4.646       3.171
x9            -0.0926      0.417     -0.222      0.824      -0.909       0.724
==============================================================================
```
Prediction on the result
```ruby
predictions = logmodel.predict(X_test)
class_prediction = [1 if x>=0.5 else 0 for x in predictions]
```
```
predictions
```
_array([0.74451727, 0.98054131, 0.06441659, 0.31870749, 0.34216635,
       0.99986554, 0.83417788, 0.09142223, 0.84015102])_
```
class_prediction
```
_[1, 1, 0, 0, 0, 1, 1, 0, 1]_

Print out the accuracy and confusion matrix
```
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, class_prediction))
print(confusion_matrix(y_test, class_prediction))
```
_0.6666666666666666_
_[[2 1]
 [2 4]]_

Create plot to visualize the AUC (Area under curve)
```
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
```
```
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
```
```ruby
plt.plot(fpr, tpr, label='ROC Curve (area = %0.3f)' %roc_auc)
plt.title('ROC curve (area = %0.3f)' %roc_auc)
plt.xlabel('FP rate')
plt.ylabel('TP rate')
```
<img width="587" alt="Screenshot 1402-11-17 at 17 03 21" src="https://github.com/kellyhuanyu/Machine-Learning-Theory/assets/105426157/240d1b96-841a-4f76-9d0b-a7b03d112c13">












