# Projects in Machine Learning

## Big mart daily data - Sales prediction  

### The Problem Statement

The data experts at BigMart have collected sales data for more than 1000 products across 10 different stores in various cities. The aim is to build a predictive model to find out the sales of each product at a particular store. Create a model by which BigMart can analyse and predict the outlet production sales

### Learning:

It is the perfect project for learning Data Analytics. The project helps the user to know 
- Data Exploration
- Data Cleaning
- Feature Engineering
- Creating Models


## Exploratory Data Analysis Project - Predicting the Quality of Wine

### The Problem Statement

In this project we need to predict the quality of Red wine in a simple non technical way. We would start with exploring the attributes and then doing outlier detection and elimination basis their requirement. 

### Learning:

- This project helps to dig deeper and helps in understanding about the two broad kind of projects namely Regression and Classification.
- How to evaluate the models and figure the best one to implement.
- Splitting the data for testing and Training.
- Baselining and optimization

## Simple Linear Regression to predict the time when Geyser will excite - Old Faithful Geyser Dataset

### The Problem Statement

The scientists, geologists and government authorities have the interest and have engaged in the observations and analysis of the geysers and eruptions throughout several decades. Old Faithful is one of the most studied and observed geysers although it is not the largest geyser active currently. In this approach, Old Faithful Geyser data is being analyzed with a machine learning approach to support the observations for predicting future behavior of not only geysers, but also the Earth.

• The dataset of Old Faithful that has been occupied here was obtained from:

https://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat

https://app.quadstat.net/dataset/r-dataset-faithful

• The dataset contains waiting time between two consecutive eruptions (in minutes) and the duration of the eruption (in minutes) for the Old Faithful geyser.

• There are 272 observations of eruptions occurred in Old Faithful Dataset.

### Learning:

In this project we get to know about the implementation of K Mean clustering and figruing out the exact frequency at which the geyser gets the next eruption.

It can be drawn that there are two features to be considered:
1. Waiting time between two consecutive eruptions (integer)
2. Duration of the eruption (floating-point)

#### K-means Clustering
K-Means is a clustering algorithm which is rapidly used from the beginning of Machine Learning era to facilitate clustering analysis of data. It is a centroid-based, iterative algorithm that partitions data into K number of non-overlapping subgroups. This algorithm is randomly initialized, and then it iterates while assigning centroids, clustering points around centroids and comparing each data point with every centroid to find the difference. This difference is measured with Euclidean distance between considered points in the calculation.

#### Applying K-means to geyser eruptions
This implementation is performed using Python programming language and related libraries to achieve the task. All 272 observations from the source have been taken into account for the purpose.

The value that has been assigned for K is 2 to perform K-means algorithm, and two clusters have been derived from the standardized data. The standardizing of data means containing data with a zero mean and standard deviation of one which is recommended because the features might not be in the same measurement units.

The two clusters in data interprets that there are two series of eruptions in Old Faithful geyser; eruptions with short intervals and eruptions with long intervals (more than 3 minutes). The eruptions with long intervals last longer than short interval eruptions, because longer eruptions require more effort than short interval discharges. Furthermore, the geyser is having an increasing number of long eruptions than shorter eruptions.

According to the above details, it can be assumed that Old Faithful geyser has varying behavior upon eruption in different situations. These conditions including atmospheric temperature, availability of water, wind speed, depth of the conduit, distant earthquakes should be analyzed further for authenticating those variations. The approach that is implemented in this scenario with K-means could provide predictions for future eruptions in terms of their duration and waiting time.

## Linear Regression project - Multiple Linear Regression using Gradient Descent method

### The Problem Statement


### Learning:


## Logistic Regression project - IRIS Dataset predicting the plant 
## KNN Implementation project on IRIS Dataset - Predict the plant with its parameters
## Insurance subscription prediction using SVC, KNN and Logistic Regression's F1 scores





