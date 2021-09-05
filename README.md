# Mode-wise Tensor Decomposition

This repository is for our paper:

[1] HanQin Cai, Keaton Hamm, Longxiu Huang, and Deanna Needell. <a href=https://jmlr.org/papers/v22/21-0287.html>Mode-wise Tensor Decompositions: Multi-dimensional Generalizations of CUR Decompositions</a>. *Journal of Machine Learning Research*, 22.185: 1-36, 2021.

###### To display math symbols properly, one may have to install a MathJax plugin. For example, [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en).


## Introduction
In this work, we generalize CUR decompositions to high-order tensors under the low-multilinear-rank setting.  
We provide two verisons of this generalization, namely Chidori and Fiber CUR decompositions.  


## Environment
This repo is developed with <a href=https://gitlab.com/tensors/tensor_toolbox/-/releases/v3.1>Tensor Toolbox v3.1</a>. A future verison of this toolbox may also increase the peformance of our code; however, we cannot guarantee their compatibility.


## Syntex

#### Chidori CUR
```
[Core,X_sub_mat] = Chidori_CUR(X, R, const);
```

#### Chidori CUR
```
[Core, X_sub_mat] = Fiber_CUR(X, R, const_R, const_C);
```

## Input Description
1. X : Inputed tensor. 
1. R : Targeted multilinear rank.
1. const : Sampling constant in **Chidori CUR**. (Default value: 2)
1. const_R : Sampling constant for core tensor in **Fiber CUR**. (Default value: 2)
1. const_C : Sampling constant for {C_i} in Fiber CUR. (Default value: 4)

* See paper for the details of constant selection.

## Output Description
1. Core : Core tensor, i.e., $\mathcal{R}$.
1. X_sub_mat : Fiber CUR components, i.e., {$C_i U_i^\dagger$}.

#### To obtain the full estimated tensor, call 
```
X_est = tensor(ttensor(Core,X_sub_mat));
```

## Demo

Clone the codes and run `test_tensor_CUR.m` for a test demo.
