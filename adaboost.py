import numpy as np 
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
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
		data = np.asarray(data)
		np.random.shuffle(data)

	X_train = (np.asarray(data))[:, 6:70].astype(np.float)
	y_train = (np.asarray(data))[:, 382].astype(np.float)

	with open('test_2012.csv', 'r') as srcfile:
		data_iter = csv.reader(srcfile, quotechar = '"')
		data = [data for data in data_iter]
		data = data[1:]

	X_test = (np.asarray(data))[:, 6:70].astype(np.float)
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
	numTwos_predicted = 0
	numTwos_correct = 0
	for i in range(len(predicted_outputs)):
		predicted = predicted_outputs[i]
		correct = correct_outputs[i]
		if correct == 2:
			numTwos_correct += 1
		if predicted == 2:
			numTwos_predicted += 1
		if predicted != correct:
			incorrect += 1
	print "number of twos in correct outputs: ", numTwos_correct
	print "number of twos in predicted outputs: ", numTwos_predicted
	return incorrect/len(inputs)

# read in the data and normalize the inputs
X_train,Y_train, X_Test = readData()
# split the training data into a training and a validation set
val_set = X_train[0:10000,]
val_out = Y_train[0:10000]
train_set = X_train[10000:len(X_train), ]
train_out = Y_train[10000:len(X_train)]
norm_packet = normalize(train_set)

# normalize the training data; normalize validation and test data 
# with the same mean and standard deviation used for training data
norm_train = norm_packet[1]
scaler = norm_packet[0]
norm_val = scaler.transform(val_set)
norm_test = scaler.transform(X_Test)

# run adaboost on the training data and check the resulting model's 
# performance on the validation set
estimatorQuantities = np.arange(50.0, 111.0 , 10.0)
learningRates = estimatorQuantities/100
n = len(learningRates) - 1
validationErrors = np.zeros(len(estimatorQuantities))
trainingErrors = np.zeros(len(estimatorQuantities))
clf = AdaBoostClassifier(n_estimators = 70, learning_rate = 0.90)
clf.fit(norm_train, train_out)
print clf.score(norm_train, train_out)
print clf.score(norm_val, val_out)
E_in = error(clf, norm_train, train_out)
E_val = error(clf, norm_val, val_out)

# keep track of output
test_outputs = np.zeros([len(norm_test), 2])

for i in range(len(norm_test)):
	test_outputs[i][0] = i
	test_outputs[i][1] = clf.predict(np.reshape(norm_test[i], (1,-1)))
np.savetxt("adaboost_submission_2012.csv", test_outputs, fmt = "%d", delimiter = ",")

print "E_in: ", E_in
print "E_val: ", E_val