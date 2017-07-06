# SAS
This is a Python realization of **S**moothed **A**lternating Least **S**quares (SAS) algorithm.

## Introduction

Based on Netflix Prize-winning algorithm, ALS, we propose SAS, an missing data recovery algorithm that is more suitable for time-series data.

SAS is designed to recover missing values for large-scale time-series data. Instead of estimating the missing values either column by column or row by row, as in traditional imputation methods, SAS recovers the missing values by extracting patterns of data based on observed entries and then estimating the missing entries with the patterns. SAS fully utilizes the information in the data to learn the pattern and make inferences, which guarantees the estimation accuracy.

