# Exact Statistical Inference for the Wasserstein Distance by Selective Inference

This package implements an exact statistical inference for the Wasserstein Distance by Selective Inference.

## Simple Demonstration 
#### For simple demonstration example without the need of installing any package, please see the following .html file 
 - 1_SIMPLE_DEMONSTRATION.html


## Installation & Requirements

This package has the following requirements:

- [numpy](http://numpy.org)
- [mpmath](http://mpmath.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://www.scipy.org)

We recommend to install or update anaconda to the latest version and use Python 3
(We used Python 3.8.3).

**NOTE: We use scipy package to solve the linear program (simplex method). However, the default package does not return the set of basic variables. Therefore, we slightly modified the package so that it can return the set of basic variables by replacing the two files '_linprog.py' and '_linprog_simplex.py' in scipy.optimize module with our modified files in the folder 'file_to_replace'.**

**NOTE: Please follow the above replacing step. Otherwise, you can not run the code.**

The results will be saved in the folder "./results" and some results are shown on the console. 

## Examples

#### (1) To reproduce the results in 1_SIMPLE_DEMONSTRATION.html, please refer to the following jupyter notebook file:
```
ex0_simple_demonstration.ipynb
```

#### (2) Checking the uniformity of the pivotal quantity in Equation (17) in the paper

To check the uniformity of the pivotal quantity, please run  
```
>> python ex1_uniform_pivot.py
```
This is another way to check the correctness of the proposed method. The pivotal quantity in Equation (17) of the paper must follow uniform distribution. 

#### (3) Example of computing CI
```
>> python ex2_confidence_interval.py
```
