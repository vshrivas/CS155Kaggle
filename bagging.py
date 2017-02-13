import numpy as np 
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from matplotlib import pyplot as plt

# read in data from the training csv
# return a tuple containing an input matrix and an output vector for the train data
# followed by the input matrix for the test data
def readData():
	# read in the data and normalize the inputs
	# load data from CSV files
	# extract data into data_array by row, first row is column labels
	with open('train_2008.csv', 'r') as srcfile:
		data_iter = csv.reader(srcfile, quotechar = '"')
		data = [data for data in data_iter]
		data = data[1:]

	X_train = (np.asarray(data))[:, 3:382].astype(np.float)
	y_train = (np.asarray(data))[:, 382].astype(np.float)

	with open('test_2008.csv', 'r') as srcfile:
		data_iter = csv.reader(srcfile, quotechar = '"')
		data = [data for data in data_iter]
		data = data[1:]

	X_test = (np.asarray(data))[:, 3:382].astype(np.float)
	return (X_train, y_train, X_test)

# normalize the input matrix
# returns the scaler used to normalize, will be useful for test inputs 
# also returns normalized inputs
def normalize(inputs):
	print "Normalizing inputs..."
	scaler = StandardScaler().fit(inputs)
	norm_inputs = scaler.transform(inputs)
	print "Done normalizing. "
	return (scaler, norm_inputs)

# calculate the error of prediction 
# this will be classification error because we want voter turnout to be in 
# the correct class
def error(clf, inputs, correct_outputs):
	predicted_outputs = clf.predict(inputs)
	# keeps track of the number of incorrect predictions
	incorrect = 0.
	for i in range(len(predicted_outputs)):
		predicted = predicted_outputs[i]
		correct = correct_outputs[i]
		if predicted != correct:
			incorrect += 1
	return incorrect/len(inputs)

# read in the data and normalize the inputs
X_train,Y_train, X_Test = readData()
# split the training data into a training and a validation set
val_set = X_train[0:5000,]
val_out = Y_train[0:5000]
train_set = X_train[5000:len(X_train), ]
train_out = Y_train[5000:len(X_train)]
norm_packet = normalize(train_set)

# normalize the training data; normalize validation and test data 
# with the same mean and standard deviation used for training data
norm_train = norm_packet[1]
scaler = norm_packet[0]
norm_val = scaler.transform(val_set)
norm_test = scaler.transform(X_Test)

# run a bagging classifier on this training set 
max_sample_values = np.arange(0, 1, 0.1)
n_estimators_values = range(1,30,1)
max_features_values = np.arange(0,1, 0.1)
clf = BaggingClassifier(n_estimators = 10, max_samples = 0.7, max_features = 0.8)
clf.fit(norm_train, train_out)
E_in = error(clf, norm_train, train_out)
E_val = error(clf, norm_val, val_out)

print "E_in: ", E_in
print "E_val: ", E_val