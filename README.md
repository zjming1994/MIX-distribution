# MIX-distribution
Code for calculating mixed distribution of two different components. 

## Requirements
The version of Python interpreter is 3.9.7 on win 10, 
As for some librarys or packages, we use: pandas 1.3.4, numpy 1.20.3, matplotlib 3.4.3 and scipy 1.7.1. 

## Installation 
After installing the above requirements (or Anaconda on https://www.anaconda.com/). 
Please download the codes, Excel files, and folder, and place them inside a new folder. 

## Demo
In mix_dist_two.py, we solve the proportions of two different components using simulation and realistic data. 

In function main_draw(), path_A = '6SLP--39240 events.xlsx' and path_B = '3SLP--34595 events(1).xlsx' are data of two different components, each of them contains
2 dimensions. To illustrate our method, we set the proportions as 2:8, 4:6, 6:4 and 8:2, respectively, and then plot the results in train_plot.png. 

In function main_test(), path_A = '三组分子数据/3S3FLP--37898 events.xlsx' and path_B = '三组分子数据/6S3FLP--47692 events.xlsx' are data of two different components, each of them contains 2 dimensions, path_AB = '三组分子数据/2. add 6S3FLP--containing 3S3FLP + 6S3FLP------11308 events.xlsx' is the data of their mixture with unknow proportion. main_test() will output the proportion of A and B, and then plot the results in test_plot4_6.png. 

## Run time 
Less than 10 seconds for our datasets. 















