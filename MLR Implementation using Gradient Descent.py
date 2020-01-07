# Title: MLR Implementation using Gradient Descent 
# Author: Rishabh Malhotra
# All Rights Reserved by Datum Guy & the Author

import csv
import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt

#############
## DATA IO ##
#############

def get_data(filepath):
	# Opens the file handler for the dataset file. Using variable 'f' we can access and manipulate our file anywhere in our code
	# after the next code line.
	f = open(filepath,"r")

	# Predictors Collection (or your input variable) (which in this case is just the duration of eruption)
	X1 = []
	X2 = []
	X3 = []

	# Output Response (or your output variable) (which in this case is the duration after which next eruption will occur.)
	Y = []

	# Initializing a reader generator using reader method from csv module. A reader generator takes each line from the file
	# and converts it into list of columns.
	reader = csv.reader(f)

	# Using for loop, we are able to read one row at a time.
	for row in reader:
		# Ignoring the Column Names Row (Header)
		if not row[3].isalpha():
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			X3.append(float(row[2]))
			Y.append(float(row[3]))

	# Close the file once we have succesffuly stored all data into our X and Y variables.
	f.close()
	return [[np.array(X1),np.array(X2),np.array(X3)],np.array(Y)]

# Normalizing Data - to make the Gradient Descent Process efficient and fast.
def normalize_data(X):
	_X = []
	for i in range(0,3):
		_X.append((X[i] - np.mean(X[i]))/(np.std(X[i])))
	return _X

#####################
## RSS Calculation ##
#####################

def RSS(x, y, betas):
	rss = 0
	for i in range(x[0].shape[0]):
		predicted_value = (betas[0] + (betas[1] * x[0][i]) + (betas[2] * x[1][i]) + (betas[3] * x[2][i]))
		actual_value = y[i]
		rss = rss + ((predicted_value - actual_value)**2)     
	return (np.sqrt(rss/x[0].shape[0]))

def predicted_value_for_ithRow(X,i,betas):
	return (betas[0] + (betas[1]*X[0][i]) + (betas[2]*X[1][i]) + (betas[3]*X[2][i]))

def gradientDescentAlgorithm(x, y, learning_rate):
	
	print ("Training Linear Regression Model using Gradient Descent")
	
	maximum_iterations = 5000
	
	# This flag lets the program know wether the gradient descent algorithm has reached it's converged state which means wether 
	# the algorithm was able to find the local minima (where the slope of RSS wrt your parameters beta_0 and beta_1 is zero)
	converge_status = False
	
	# num_rows stores the number of datapoints in the current dataset provided for training.
	num_rows = x[0].shape[0]

	# Initial Value of parameters ((beta_0, beta_1) - for our simple linear regression case)
	betas = [0,0,0,0]

	# Initial Error or RSS(beta_0,beta_1) based on the initial parameter values
	error = RSS(x, y, betas)
	print('Initial Value of RSS (Cost Function)=', error)
	
	# Iterate Loop
	num_iter = 0
	while not converge_status:
		# for each training sample, compute the gradient (d/d_beta j(beta))
		gradient_0 = 1.0/num_rows * sum([(predicted_value_for_ithRow(x,i,betas) - y[i]) for i in range(num_rows)]) 
		gradient_1 = 1.0/num_rows * sum([(predicted_value_for_ithRow(x,i,betas) - y[i])*x[0][i] for i in range(num_rows)])
		gradient_2 = 1.0/num_rows * sum([(predicted_value_for_ithRow(x,i,betas) - y[i])*x[1][i] for i in range(num_rows)])
		gradient_3 = 1.0/num_rows * sum([(predicted_value_for_ithRow(x,i,betas) - y[i])*x[2][i] for i in range(num_rows)])

		# Computation of new parameters according to the current gradient.
		temp0 = betas[0] - learning_rate * gradient_0
		temp1 = betas[1] - learning_rate * gradient_1
		temp2 = betas[2] - learning_rate * gradient_2
		temp3 = betas[3] - learning_rate * gradient_3
		
	
		# Simultaneous Update of Parameters betas.
		betas[0] = temp0
		betas[1] = temp1
		betas[2] = temp2
		betas[3] = temp3

		
		current_error = RSS(x, y, betas)
		
		if num_iter % 250 == 0:
			print ('Iteration',num_iter+1,'Current Value of RSS (Cost Function) based on updated values of beta parameters = ', current_error)
		
		# Automatic Stopping of Gradient Descent based on the change in the cost from the previous step to the current step. It compares the error (cost) and if there
		# is no significant change it stops the process.
		if (error - current_error)/(error)<0.00000000000000001:
			print("No Significant change in learning. Stopping Gradient Descent Process")
			break
			
		error = current_error   # update error 
		num_iter = num_iter + 1  # update iter
	
		if num_iter == maximum_iterations:
			print ("Training Interrupted as Maximum number of iterations were crossed.\n\n")
			converge_status = True

	return (betas)

# Method to predict response variable Y for new values of X  using the estimated coefficients.
# This method can predict Response variable (Y) for single as well as multiple values of X. If only a single numerical Value
# input variable (X). It will return the prediction for only that single numerical
# value. If a collection of different values for input variable (list) is passed, it will return a list of predictions
# for each input value.
# "if" statement on line number 72 takes care of understanding if the input value is singular or a list.
def predict(coef,X):
	beta_0 = coef[0]
	beta_1 = coef[1]
	beta_2 = coef[2]
	beta_3 = coef[3]
	
	fy = []
	if len(X) > 1:
		for x in X:
			fy.append(beta_0 + (beta_1 * x[0]) + (beta_2 * x[1]) + (beta_3 * x[2]))
		return fy

	# Our Regression Model defined using the coefficients from slr function
	x = X[0]
	Y = beta_0 + (beta_1 * x[0]) + (beta_2 * x[1]) + (beta_3 * x[2])

	return Y

# Change the File Path Accordingly.
X,Y = get_data("../Data/50_Startups.csv")
X = normalize_data(X)

################################################
## Model Training (or coefficient estimation) ##
################################################
# Using our gradient descent function we estimate coefficients of our regression line. The gradient descent function 
# returns a list of coefficients

coefficients = gradientDescentAlgorithm(X,Y,0.05)

########################
## Making Predictions ##
########################

# Using our predict function and the coefficients given by our slr function we can now predict the time it will take
# for the next eruption.
print ("Final Values for Beta Parameters are (from beta_0 to beta_3) :",coefficients)
print (predicted_value_for_ithRow(X,3,coefficients))

############################
## Performance Evaluation ##
############################

print ("\n\nAccuracy Metrics of the model\n-------------------------------------")

# Calculation of RSE
RSS = 0
X = np.transpose(X)
for idx in range(0,len(X)):
	actual_y = Y[idx]
	predicted_y = predict(coefficients,[X[idx,0:6]])
	RSS = RSS + ((actual_y - predicted_y)**2)
RSE = np.sqrt((1/float(X.shape[0]-2))*RSS)

print ("Residual Standard Error:",RSE)
print ("% Residual Standard Error (over average Interval):", (RSE/np.mean(Y))*100)


# Calculation of R_Squared
TSS = 0
for idx in range(0,len(X)):
	actual_y = Y[idx]
	TSS = TSS + ((actual_y - np.mean(Y))**2)
R_Squared = ((TSS) - (RSS)) / (TSS)

print ("\nR-Squared Value:",R_Squared)