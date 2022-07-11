---
title: "Machine Learning Notes"
author: "Xuduo Gu"
header-includes:
   - \usepackage{bm}
   - \usepackage{amsmath}
---

## Regression

Given the input $X=\begin{bmatrix}
    \boldmath{x_1} \\
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
\end{bmatrix}$, we want to find the parameter $\theta = (\textbf{w}, b)$
