# SAS
This is a Python realization of **S**moothed **A**lternating Least **S**quares (SAS) algorithm.

## Introduction

Based on Netflix Prize-winning algorithm, ALS, we propose SAS, an missing data recovery algorithm that is more suitable for time-series data.

SAS is designed to recover missing values for large-scale time-series data. Instead of estimating the missing values either column by column or row by row, as in traditional imputation methods, SAS recovers the missing values by extracting patterns of data based on observed entries and then estimating the missing entries with the patterns. SAS fully utilizes the information in the data to learn the pattern and make inferences, which guarantees the estimation accuracy.

SAS solves the matrix completion problem by implementing low-rank matrix decomposition. Specifically, it assumes there are certain patterns in the data, and that these patterns are described in a low-rank matrix. Apart from the low-rank assumption, SAS considers the smoothness of the data, which helps the algorithm better suit time-series data.

## Usage

To implement SAS in data recovery tasks, one can simply download the *sas.py* file and put it in the working directory. Then import the SAS class with the following command.
`from sas import SAS`
