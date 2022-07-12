---
title: "Machine Learning Notes"
author: "Xuduo Gu"
---

## Regression

### Linear Regression

#### Alogrithm for Linear Regression

Given the input $X=\begin{bmatrix}
    \mathbf{x_1} \\
    \vdots \\
    \mathbf{x_n}
\end{bmatrix} =
\begin{bmatrix}
    x_{11} & \dots  & x_{1k} \\
    \vdots & \ddots & \vdots \\
    x_{n1} & \dots  &x_{nk}
\end{bmatrix}$ with the corresponding target value $Y=\begin{bmatrix}
    y_{1} \\
    \vdots \\
    y_{n}
\end{bmatrix}$, we want to find the parameter $\theta = (\textbf{w}, b)$ so that $f(\mathbf{x_i}) = \mathbf{xw+b} = \hat{y_i} \approx y_i$.  

Make $ X'=\begin{bmatrix}
    x_{11} & \dots  & x_{1k} & 1\\
    \vdots & \ddots & \vdots & \vdots\\
    x_{n1} & \dots  &x_{nk}  & 1\\
\end{bmatrix} $ and $\mathbf{w'}=
\begin{bmatrix}
    \mathbf{w_1} \\
    \vdots \\
    \mathbf{w_k} \\
    1
    \end{bmatrix}$

Cost function: $L(\theta) = (Y-X'\mathbf{w'})^T(Y-X'\mathbf{w'})$.  

Taking derivative of $L$ over $\mathbf{w'}$ to find minimum loss, we have:  

$\frac{\delta L}{\delta \mathbf{w'}} = 2X'^T(Y-X'\mathbf{w'}) = \mathbf{0}  \\
\Rightarrow X'^TY = X'^TX'\mathbf{w'} \\
\Rightarrow \mathbf{w'} = ( X'^TX)^{-1}X'^TY$

### Regression Tree

#### Alogrithm for Regression Tree

Given the input $X=\begin{bmatrix}
    \mathbf{x_1} \\
    \vdots \\
    \mathbf{x_n}
\end{bmatrix} =
\begin{bmatrix}
    x_{11} & \dots  & x_{1k} \\
    \vdots & \ddots & \vdots \\
    x_{n1} & \dots  &x_{nk}
\end{bmatrix}$ with the corresponding target value $Y=\begin{bmatrix}
    y_{1} \\
    \vdots \\
    y_{n}
\end{bmatrix}$, we want to build a regression tree.

The goal is to find a threshold for each internal node such that the split of data generates the minimum loss. So we want to find $argmin_{s, d, c_1, c_2}\sum_{i=1}^{n}(y_i-c_1)^2\mathbf{1}\{x_{id}\leq s\}+(y_i-c_2)^2\mathbf{1}\{x_{id} > s\}$.

Cost function: $L(s, d, c_1, c_2) = \sum_{i=1}^{n}(y_i-c_1)^2\mathbf{1}\{x_{id}\leq s\}+(y_i-c_2)^2\mathbf{1}\{x_{id} > s\}$.

With fixed $s$ and $d$, taking derivative of $L$ over $c_1$ and $c_2$, we have :

$\frac{\delta \sum_{i=1}^{n}(y_i-c_1)^2\mathbf{1}\{x_{id}\leq s\}+(y_i-c_2)^2\mathbf{1}\{x_{id} > s\}}{\delta c_1} = -2\sum_{i=1}^{n}(y_i-c_1)\mathbf{1}\{x_{id}\leq s\}=0\Rightarrow c_1=\frac{1}{|\{x_i|x_{id}\leq s\}|}\sum_{i=1}^{n}y_i\mathbf{1}\{x_{id}\leq s\}$

$\frac{\delta \sum_{i=1}^{n}(y_i-c_1)^2\mathbf{1}\{x_{id}\leq s\}+(y_i-c_2)^2\mathbf{1}\{x_{id} > s\}}{\delta c_2} = -2\sum_{i=1}^{n}(y_i-c_2)\mathbf{1}\{x_{id}> s\}=0\Rightarrow c_2=\frac{1}{|\{x_i|x_{id}>s\}|}\sum_{i=1}^{n}y_i\mathbf{1}\{x_{id}> s\}$

For $s$ and $d$, we can loop over all $k$ features of all $n$ inputs to find the smallest value.

#### Analysis for Regression Tree

Each internal node loops over $nk$ different kinds of seperation, and each seperation loops over $n$ target values to calculate the loss. So complexity for each node is $O(nk^2)$. Meanwhile, considering there are $O(n)$ internal nodes, **the total complexity for training is $O(n^2k^2)$**. To estimate value for each test data point, the **complexity for testing is $O(logn)$**.

Tree models such as regression tree do not need rescaling, while non-tree models such as linear regression may need it. This is because rescaling does not change the way tree models split values, meanwhile, it could reduce the number of iterations for those iterative models (e.g. K-Means).

**Advantage**:

1. Does not require normalization and rescaling.
2. Does not need feature engineering.
3. Can deal with missing values of features.
4. Fast in predicting.

**Disadvantages**:

1. Not smooth, can only return certain values.
2. Not suitable for sparse data with high dimensionality.

### Boosting Regression Tree

#### Algorithm for Boosting Regression Tree

Using AdaBoost to improve regression tree:

1. Initialize the function $f_0(x)=0$.

2. For $m=1 \dots M$, where $M$ is the number of stumps:

   1. Calculate $r_{im} = y_i-f_{m-1}(x_i)$.

   2. Find a stump $C_m$ that seperates data in the way that generates the minimum loss w.r.t $r_{im}$. That is, to find $argmin_{s, d, c_1, c_2}\sum_{i=1}^{n}(r_{im}-c_1)^2\mathbf{1}\{x_{id}\leq s\}+(r_{im}-c_2)^2\mathbf{1}\{x_{id} > s\}$.

        Here we have:
        $$C_m(x_i) = \begin{equation*}
        \left\{
        \begin{align}
        c_1, \text{if $x_{id} \leq s$}\\
        c_2, \text{if $x_{id} > s$}\\
        \end{align}
        \right.
        \end{equation*}$$

   3. $f_m(x)=f_{m-1}(x) + C_m(x)$

3. Return the boosting regression tree $f(x)=\sum_{m=1}^M C_m(x)$.

#### Analysis for Boosting Regression Tree

This model is an additive model.

See [Analysis for Boosting](#Analysis_for_Boosting)

### Gradient Boosting Regression Tree

#### Algorithm for Gradient Boosting Regression Tree

1. Initialize $f_0(x)=argmin_c\Sigma_{i=1}^N(y_i-c)^2 = \frac{1}{N}\Sigma_{i=1}^Ny_i$.
2. For $m=1\dots M$, where $M$ is the number of regression trees.
   1. For $i=1\dots N$, calculate $r_{mi}=-\frac{\delta L(y_i, f_{m-1}(x_i))}{\delta f_{m-1}(x_i)} = -\frac{\delta (y_i-f_{m-1}(x_i))^2}{\delta f_{m-1}(x_i)} = 2(y_i-f_{m-1}(x_i))$
   2. Build a regression tree to fit $r_m$.
   3. For each of the $J$ leaf nodes, the corresponding value is $c_{mj} = argmin_c\Sigma_{x_i\in node_j}L(y_i, f_{m-1}(x_i)+c)$.
   Since we want $\frac{\delta \Sigma_{x_i\in node_j}L(y_i, f_{m-1}(x_i)+c)}{\delta c} = \frac{\delta \Sigma_{x_i\in node_j}(y_i-f_{m-1}(x_i)-c)^2}{\delta c}=2\Sigma_{x_i\in node_j}(c+f_{m-1}(x_i)-y_i)=0$.
   Therefore, $c_{mj}=\frac{1}{|node_j|}\Sigma_{x_i\in node_j}(y_i-f_{m-1}(x_i))$.
   4. Update $f_m(x) = f_{m-1}(x)+\Sigma_{i=1}$.
3. Return GBDT regression tree $f(x)=f_0(x)+\Sigma_{m=1}^M\Sigma_{j=1}^Jc_{mj}\mathbf{1}\{x\in node_j\}$.

#### Analysis for Gradient Boosting Regression Tree

GBDT regression tree is similar to boosting regression tree when the loss function $L$ is the least square error. However, least square error is susceptible to outliers, therefore some other loss functions could be used (e.g. Huber loss). In such cases, the GBDT regression tree is no longer similar to boosting regression tree since it fits a different set of $r_{mi}$.

**Shrinkage** is a skill used when an learning rate $0<\alpha<1$ is applied to the new week learner in each iteration. Smaller learning rate generally results in better testing error, however, it requires more weak learners. In practice, a small learning rate (e.g. $\alpha=0.1$) is combined with early stopping technique to determine the total number of week learners.

**More to go**: XGboost, LightGBM.

## Classification

### Logistic Regression  

### Decision Tree

#### Analysis for Decision Tree

**Advantages**:

1. Does not require normalization and rescaling.
2. Does not need feature engineering.
3. Can deal with missing values of features.
4. Fast in predicting.

**Disadvantages**:

1. Not suitable for sparse data with high dimensionality.
2. Not smooth.
3. Easily overfit if used singly.

### Random Forest

## Ensemble

### Boosting

#### Analysis for Boosting

**AdaBoost does not overfit easily.** In many cases, when the training error approaches 0, the testing error/generalized error will increase. However, for AdaBoost, even when the training error is reduced to 0, the generalized error will continue to decrease. AdaBoost increases confidence if training continues after the training error is 0. This is because of the boosting margin, which bounds the generalized error. More [here](https://jeremykun.com/2015/09/21/the-boosting-margin-or-why-boosting-doesnt-overfit/).