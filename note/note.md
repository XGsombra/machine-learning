---
title: "Machine Learning Notes"
author: "Xuduo Gu"
---

## Regression

### Linear Regression

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

Make $X'=\begin{bmatrix}
    x_{11} & \dots  & x_{1k} & 1\\
    \vdots & \ddots & \vdots & \vdots\\
    x_{n1} & \dots  &x_{nk}  & 1
\end{bmatrix}$ and $\mathbf{w'}=
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