# Projects in Machine Learning

#### 1) Big mart daily data - Sales prediction  
#### 2) Exploratory Data Analysis Project - Predicting the Quality of Wine
#### 3) Simple Linear Regression to predict the time when Geyser will excite - Old Faithful Geyser Dataset
#### 4) Linear Regression project - Multiple Linear Regression using Gradient Descent method
#### 5) Logistic Regression project - IRIS Dataset predicting the plant 
#### 6) KNN Implementation project on IRIS Dataset - Predict the plant with its parameters
#### 7) Insurance subscription prediction using SVC, KNN and Logistic Regression's F1 scores



## 1) Big mart daily data - Sales prediction  

### The Problem Statement

The data experts at BigMart have collected sales data for more than 1000 products across 10 different stores in various cities. The aim is to build a predictive model to find out the sales of each product at a particular store. Create a model by which BigMart can analyse and predict the outlet production sales

### Learning:

It is the perfect project for learning Data Analytics. The project helps the user to know 
- Data Exploration
- Data Cleaning
- Feature Engineering
- Creating Models


## 2) Exploratory Data Analysis Project - Predicting the Quality of Wine

### The Problem Statement

In this project we need to predict the quality of Red wine in a simple non technical way. We would start with exploring the attributes and then doing outlier detection and elimination basis their requirement. 

### Learning:

- This project helps to dig deeper and helps in understanding about the two broad kind of projects namely Regression and Classification.
- How to evaluate the models and figure the best one to implement.
- Splitting the data for testing and Training.
- Baselining and optimization

## 3) Simple Linear Regression to predict the time when Geyser will excite - Old Faithful Geyser Dataset

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

## 4) Linear Regression project - Multiple Linear Regression using Gradient Descent method

### The Problem Statement

In this project you need to find the  loss function using the Gradient descent method and evaluate the exact point or range where the loss is minimum.

### Learning:

In this project you can learn how the gradient descent algorithm works and implement it from scratch in python. First we look at what linear regression is, then we define the loss function. We learn how the gradient descent algorithm works and finally we will implement it on a given data set and make predictions.

#### Loss Function
The loss is the error in our predicted value of m and c. Our goal is to minimize this error to obtain the most accurate value of m and c.
We will use the Mean Squared Error function to calculate the loss. 

#### The Gradient Descent Algorithm
Gradient descent is an iterative optimization algorithm to find the minimum of a function. Here that function is our Loss Function.

##### Understanding Gradient Descent
Imagine a valley and a person with no sense of direction who wants to get to the bottom of the valley. He goes down the slope and takes large steps when the slope is steep and small steps when the slope is less steep. He decides his next position based on his current position and stops when he gets to the bottom of the valley which was his goal.

Gradient descent is one of the simplest and widely used algorithms in machine learning, mainly because it can be applied to any function to optimize it. Learning it lays the foundation to mastering machine learning.


### Summary
In this project you will discover the simple linear regression model and how to train it using stochastic gradient descent.

You work through the application of the update rule for gradient descent. You also learned how to make predictions with a learned linear regression model.



## 5) Logistic Regression project - IRIS Dataset predicting the plant 

### The Problem Statement

The iris dataset contains the following data
- 50 samples of 3 different species of iris (150 samples total)
- Measurements: sepal length, sepal width, petal length, petal width
- The format for the data: (sepal length, sepal width, petal length, petal width)

We need to be able to predict the sample given using the characteristics studied of the data we have in hand.

### Implementation

1) Logistic Regression from scratch
- use a sigmoid function to output a result between 0 & 1
      -- return 1 / (1 + np.exp(-z))
- use a loss function with parameters (weights - theta) to compute the best value for them
      ## initially pick random values
      ## return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
- gradient descent
      gradient = np.dot(X.T, (h - y)) / y.shape[0]
- predictions
      def predict_probs(X, theta): return sigmoid(np.dot(X, theta))

2) Logistic Regression from scikit-learn
      docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
      much faster than the scratch implementation



### Learning:



## 6) KNN Implementation project on IRIS Dataset - Predict the plant with its parameters




The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.

It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

The columns in this dataset are:

Id
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
Species
## 7) Insurance subscription prediction using SVC, KNN and Logistic Regression's F1 scores





