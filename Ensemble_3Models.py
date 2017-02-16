import csv
import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import scipy as sp

def Ensemble():
	print ('ensemble method called')
	X_train, y_train, X_val, y_val, X_test, numFeaturesDel = loadData()

	trainingAccuracyRF, validationAccuracyRF, valPredictionsRF, testPredictionsRF = RandomForests(X_train, y_train, X_val, y_val, X_test)
	#print (trainingAccuracyRF, validationAccuracyRF, valPredictionsRF, testPredictionsRF)

	trainingAccuracyDT, validationAccuracyDT, valPredictionsDT, testPredictionsDT = DecisionTree(X_train, y_train, X_val, y_val, X_test)
	#print (trainingAccuracyDT, validationAccuracyDT, valPredictionsDT, testPredictionsDT)

	trainingAccuracyNN, validationAccuracyNN, valPredictionsNN, testPredictionsNN = NeuralNets(X_train, y_train, X_val, y_val, X_test, numFeaturesDel)
	#print (trainingAccuracyNN, validationAccuracyNN, valPredictionsNN, testPredictionsNN)

	print validationAccuracyRF
	print validationAccuracyDT
	print validationAccuracyNN

	correctPredictions = 0.0
	# generate average validation predictions and compare with actual predictions
	for i in range(0, len(y_val)):
		numZeroPredicts = 0.0
		numOnePredicts = 0.0

		if(valPredictionsRF[i] == 0):
			numZeroPredicts += validationAccuracyRF
		else:
			numOnePredicts += validationAccuracyRF

		if(valPredictionsDT[i] == 0):
			numZeroPredicts += validationAccuracyDT
		else:
			numOnePredicts += validationAccuracyDT

		if(valPredictionsNN[i] == 0):
			numZeroPredicts += validationAccuracyNN[1]
		else:
			numOnePredicts += validationAccuracyNN[1]

		#print numZeroPredicts
		#print numOnePredicts

		finalPrediction = 1
		if(numZeroPredicts > numOnePredicts):
			finalPrediction = 0

		if(finalPrediction == y_val[i]):
			correctPredictions += 1

	print('ensemble accuracy:', correctPredictions / len(y_val))

	test_outputs = np.zeros([len(X_test), 2])
	for i in range(0, len(X_test)):
		numZeroPredicts = 0.0
		numOnePredicts = 0.0

		if(testPredictionsRF[i] == 0):
			numZeroPredicts += validationAccuracyRF
		else:
			numOnePredicts += validationAccuracyRF

		if(testPredictionsDT[i] == 0):
			numZeroPredicts += validationAccuracyDT
		else:
			numOnePredicts += validationAccuracyDT

		if(testPredictionsNN[i] == 0):
			numZeroPredicts += validationAccuracyNN[1]
		else:
			numOnePredicts += validationAccuracyNN[1]

		finalPrediction = 1
		if(numZeroPredicts > numOnePredicts):
			finalPrediction = 0

		test_outputs[i][0] = i
		test_outputs[i][1] = finalPrediction + 1


	np.savetxt("test_submission_ensemble.csv", test_outputs, fmt = "%d", delimiter = ",")


# normalize the input matrix
# returns the scaler used to normalize, will be useful for test inputs 
# also returns normalized inputs
def normalize(inputs):
	print "Normalizing inputs..."
	scaler = StandardScaler().fit(inputs)
	norm_inputs = scaler.transform(inputs)
	print "Done normalizing. "
	return norm_inputs

def loadData():
	# load data from CSV files
	# extract data into data_array by row, first row is column labels
	with open('train_2008.csv', 'r') as srcfile:
		data_iter = csv.reader(srcfile, quotechar = '"')
		data = [data for data in data_iter]
		data = data[1:]

	X = (np.asarray(data))[:, 3:382].astype(np.float)
	y = (np.asarray(data))[:, 382].astype(np.float)

	# change labels to 0's and 1's instead of 1's and 2's
	y = np.subtract(y, 1)

	with open('test_2008.csv', 'r') as srcfile:
		data_iter = csv.reader(srcfile, quotechar = '"')
		data = [data for data in data_iter]
		data = data[1:]

	X_test = (np.asarray(data))[:, 3:382].astype(np.float)

	numFeaturesDel = 0
	# if standard deviation of column is less than some threshold, delete column
	print X.shape
	print X[:, 340]
	stdThreshold = 0.05
	correlationThreshold = 0.03

	for i in reversed(range(0, 378)):
		colStd = np.std(X[:, i])
		colCorrelation = sp.stats.pearsonr(X[:, i], y)[0]
		# if column has a small std (no variation for different samples)
		# or the column's correlation with the labels is low (column data not important for predicting label)
		# then delete the column
		if (colStd < stdThreshold) or (colCorrelation < correlationThreshold):
			numFeaturesDel += 1
			X = np.delete(X, i, 1)
			X_test = np.delete(X_test, i, 1)

	print numFeaturesDel

	# normalize X
	X = normalize(X)

	# normalize X test
	X_test = normalize(X_test)

	# shuffle samples
	np.random.shuffle(X)

	# split the training data into a training and a validation set
	X_val = X[0:1000,]
	y_val = y[0:1000]
	X_train = X[1000:len(X), ]
	y_train = y[1000:len(X)]

	print (X_train.shape)
	print (X_val.shape)

	print (y_train.shape)
	print (y_val.shape)

	return (X_train, y_train, X_val, y_val, X_test, numFeaturesDel)


def RandomForests(X_train, y_train, X_val, y_val, X_test):
	# train model
	clf = RandomForestClassifier(n_estimators=300, max_depth=12, criterion='gini')
	clf.fit(X_train, y_train)

	# evaluate on training and validation sets
	trainingAccuracy = clf.score(X_train, y_train)
	validationAccuracy = clf.score(X_val, y_val)

	valPredictions = clf.predict(X_val)
	# generate test set predictions
	testPredictions = clf.predict(X_test)
	return (trainingAccuracy, validationAccuracy, valPredictions, testPredictions)


def DecisionTree(X_train, y_train, X_val, y_val, X_test):
	# train model
	split_size = 1700
	clf = DecisionTreeClassifier(min_samples_split = split_size, criterion = 'gini')
	clf.fit(X_train, y_train)

	# evaluate on training and validation sets
	trainingAccuracy = clf.score(X_train, y_train)
	validationAccuracy = clf.score(X_val, y_val)

	valPredictions = clf.predict(X_val)
	# generate test set predictions
	testPredictions = clf.predict(X_test)
	return (trainingAccuracy, validationAccuracy, valPredictions, testPredictions)


def NeuralNets(X_train, y_train, X_val, y_val, X_test, numFeaturesDel):
	## In your homework you should transform each input data point
	## into a single vector here and should transform the 
	## labels into a one hot vector using np_utils.to_categorical

	# our results fall into two categories
	y_train_hot = np.empty([0, 2])
	X_train_hot = X_train

	for i in range(0, y_train.size):
		y_train_hot = np.vstack((y_train_hot, to_categorical(y_train[i], 2)))


	y_val_hot = np.empty([0, 2])
	X_val_hot = X_val

	for i in range(0, y_val.size):
		y_val_hot = np.vstack((y_val_hot, to_categorical(y_val[i], 2)))

	print (X_train.shape)
	print (y_train.shape)
	print (X_train_hot.shape)
	print (y_train_hot.shape)

	## Create your own model here given the constraints in the problem
	model = Sequential()
	#model.add(Flatten())  # Use np.reshape instead of this in hw
	model.add(Dense(1000, input_shape=(379 - numFeaturesDel,), activation='relu'))
	model.add(Dropout(0.9))

	model.add(Dense(800, activation='sigmoid'))
	model.add(Dropout(0.7))

	model.add(Dense(600, activation='sigmoid'))
	model.add(Dropout(0.5))

	model.add(Dense(400, activation='sigmoid'))
	model.add(Dropout(0.3))

	model.add(Dense(200, activation='sigmoid'))
	model.add(Dropout(0.2))

	## Once you one-hot encode the data labels, the line below should be predicting probabilities of each of the 10 classes
	## e.g. it should read: model.add(Dense(10)), not model.add(Dense(1))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	## Printing a summary of the layers and weights in your model
	model.summary()

	#rmsprop and adam optimizers
	model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
	
	batchSize = 1000

	fit = model.fit(X_train_hot, y_train_hot, batch_size=batchSize, nb_epoch=10,
    	verbose=1)

	scoreTrain = model.evaluate(X_train_hot, y_train_hot, verbose=0)
	print('Train score:', scoreTrain[0])
	print('Train accuracy:', scoreTrain[1])

	scoreVal = model.evaluate(X_val_hot, y_val_hot, verbose=0)
	print('Validation score:', scoreVal[0])
	print('Validation accuracy:', scoreVal[1])

	valPredictions = model.predict_classes(X_val_hot)

	testPredictions = model.predict_classes(X_test)

	return (scoreTrain, scoreVal, valPredictions, testPredictions)

	'''# keep track of output
	test_outputs = np.zeros([len(X_test), 2])

	for i in range(len(X_test)):
		test_outputs[i][0] = i
		prediction = model.predict(np.reshape(X_test[i], (1,-1)))
		# higher probability that example falls in the first class 
		if prediction[0][0] > prediction[0][1]:
			test_outputs[i][1] = 1
		else:
			test_outputs[i][1] = 2

	np.savetxt("test_submission.csv", test_outputs, fmt = "%d", delimiter = ",")'''

Ensemble()





