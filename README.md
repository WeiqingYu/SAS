# SAS
This is a Python realization of **S**moothed **A**lternating Least **S**quares (SAS) algorithm.

## Introduction

Based on Netflix Prize-winning algorithm, ALS, we propose SAS, an missing data recovery algorithm that is more suitable for time-series data.

SAS is designed to recover missing values for large-scale time-series data. Instead of estimating the missing values either column by column or row by row, as in traditional imputation methods, SAS recovers the missing values by extracting patterns of data based on observed entries and then estimating the missing entries with the patterns. SAS fully utilizes the information in the data to learn the pattern and make inferences, which guarantees the estimation accuracy.

SAS solves the matrix completion problem by implementing low-rank matrix decomposition. Specifically, it assumes there are certain patterns in the data, and that these patterns are described in a low-rank matrix. Apart from the low-rank assumption, SAS considers the smoothness of the data, which helps the algorithm better suit time-series data.

## Usage

To implement SAS in data recovery tasks, one can simply download the *sas.py* file and put it in the working directory. Then import the SAS class with the following command.

`from sas import SAS`

Once imported the class, one can initialize the class with the following command.

`sasobject = SAS(lam=1.5,mu=2,rank=20,thres = 0.001, maxit = 20)`

Then use the command below to recover a partially observed matrix (numpy array).

`res_A, res_B = sasobject.fit(data = pomat)`


## Parameters

One can specify five parameters at class initialization, i.e., *lam*, *mu*, *rank*, *thres*, and *maxit*. 

* *lam*: The regularization term in ALS to prevent general overfitting. Default: 1.5.

* *mu*: The smoothness regularization term to penalize turbulence in the data. Default: 2.

* *rank*: The rank of decomposition. Default: 20.

* *thres*: Termination threshold for the algorithm. Default: 0.001.

* *maxit*: Maximum number of iteration. Default: 20.


