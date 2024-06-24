# LS Double Descent
This code showcases the double descent phenomenon on the Least Squares problem.  
With each step I increased the amount of parameters the model learns from the training data  
and plotted the results of the learned model on the test data.

# LS
Least squares is a classical linear regression method, that minimizes the sum of squared errors
between the points to study a trend line.  
Here is an example with points in 1 dimension (second dimension is the label) and their distance from the trend line:  

![image](https://github.com/Shahar6/DoubleDescent_LS/assets/79195545/ccf01285-c56d-49ab-ad1e-584dd1257fb4)

# This program
In this case, we generate a training set and a test set from a certain distribution (more detail below)
and we use the train set to learn a classifier with a changing amount of parameters {1,...,128}
We then plot the errors of the classifier on the test set for all the parameters and observe the double descent phenomenon!

# Generating the data
To set the stage some parameters are decided arbitrarily:  
Input dimension = 128  
output dimension = 1  
$x$ ~ $𝓝(𝟎,$ $𝚺_x)$  
$𝚺_𝐱 = 𝐂_d𝚲_x𝐂^𝑇_d$ where $𝐂_𝑑$ is the d x d Discrete Cosine Transform (DCT) matrix.  
$ϵ$ ~ $𝓝(0,$ $𝜎^2_𝜖)$  
$𝑦 = 𝐱^𝑇𝛃 + ϵ$

In the first part, I generate one training set of 32 samples, and one test set of size 1000,
calculate the classifier for each $1\le p \le128$ and plot the graph of $l_2$ error over the test set.
In the second part, I generate 500 training sets of 32 samples, and plot the average of error, along with
the bias and variance of the models.

# Calculating the predictors
Since we use between 1 and 128 parameters for the predictor, and train it on 32 samples,  
there can be either 0-1 solutions (when p <= 32), or 0/infinite amount (when p > 32) this depends on whether or not $(𝚽^𝑇𝚽)$ has an inverse  
The way to find the optimal predictor is:  
$a = (𝚽^𝑇𝚽)^{-1} * 𝚽^𝑇 * y$ if there is an inverse we use it  
$a = (𝚽^𝑇𝚽)^+ *𝚽^𝑇 *y$ if there isn't an inverse we can use the pseudo inverse  
where $𝚽=XU_p$ and $U_p$ is a dxp orthonormal matrix that converts the input from nxd to nxp  

# Results
Plotting the first part (one training dataset) we can see this error:

![image](https://github.com/Shahar6/DoubleDescent_LS/assets/79195545/679824f3-8a38-47e9-8613-2cc74d3b0281)
  
Note: because the data is from a distribution, the result of one dataset isn't reputed enough,  
each run generates different values (all follow a certain trend tho)  

And now plotting the second part (average of 500 training data sets and also variance and bias):

![image](https://github.com/Shahar6/DoubleDescent_LS/assets/79195545/d1710b6d-6873-48ba-88d1-9ded7b82d348)

In both graphs we see the double descent. In the second graph we can see that the Bias went up sharply  
from around 30 parameters, and then went down sharply at around 60, from there on, there is a small decrease  
that is the double descent and what you call - perfect fitting. :)  
It is worth stressing again that 32 was the size of the training data, that is why as we approach 32  
the training error increases dramatically. (overfit)

Here is the graph of the same run but with 1000 training sets to average off:  
![image](https://github.com/Shahar6/DoubleDescent_LS/assets/79195545/713aa398-1657-4d35-ab5f-22abac007148)

