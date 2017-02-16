import numpy as np
import csv
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.stats.stats import pearsonr
from scipy import stats
from matplotlib import pyplot as plt


# normalize the input matrix
# returns the scaler used to normalize, will be useful for test inputs
# also returns normalized inputs
def normalize(inputs):
	print ("Normalizing inputs...")
	scaler = StandardScaler().fit(inputs)
	norm_inputs = scaler.transform(inputs)
	print ("Done normalizing. ")
	return (scaler, norm_inputs)

# calculate the error of prediction
# this will be classification error because we want voter turnout to be in
# the correct class
# def error(clf, inputs, correct_outputs):
# 	predicted_outputs = clf.predict(inputs)
# 	# keeps track of the number of incorrect predictions
# 	incorrect = 0.
# 	for i in range(len(predicted_outputs)):
# 		predicted = predicted_outputs[i]
# 		correct = correct_outputs[i]
# 		if predicted != correct:
# 			incorrect += 1
# 	return incorrect/len(inputs)

# read in the data and normalize the inputs
# load data from CSV files
# extract data into data_array by row, first row is column labels
with open('train_2008.csv', 'r') as srcfile:
	data_iter = csv.reader(srcfile, quotechar = '"')
	data = [data for data in data_iter]
	data = data[1:]

# X_train = (np.asarray(data))[:, 3:382].astype(np.float)
# y_train = (np.asarray(data))[:, 382].astype(np.float)


X_train = (np.asarray(data))[:, 3:382].astype(np.float)
y_train = (np.asarray(data))[:, 382].astype(np.float)

with open('test_2012.csv', 'r') as srcfile:
	test_data_iter = csv.reader(srcfile, quotechar = '"')
	test_data = [test_data for test_data in test_data_iter]
	test_data = test_data[1:]

X_test = (np.asarray(test_data))[:, 3:382].astype(np.float)

# numFeaturesDel = 0
# for i in reversed(range(0, 378)):
# 	if np.std(X_train[:, i]) == 0 or \
# 		np.absolute(scipy.stats.pearsonr(X_train[:, i], y_train)[1]) <= 0.01:
# 		numFeaturesDel += 1
# 		X_train = np.delete(X_train, i, 1)
# 		X_test = np.delete(X_test, i, 1)
# print (numFeaturesDel)
# print(np.shape(X_train), np.shape(X_test))
#np.random.shuffle(X_train)

# split int training and validation
# val_set = X_train[len(X_train)-10000:len(X_train),]
# print(len(X_train))
# val_out = y_train[len(X_train)-10000:len(X_train)]
train_set = X_train
train_out = y_train
norm_packet = normalize(train_set)

# normalize the training data; normalize validation data with the same mean
# and standard deviation used for training data
norm_train = norm_packet[1]
scaler = norm_packet[0]
# norm_val = scaler.transform(val_set)

# normalize test input
norm_test = scaler.transform(X_test)

# keep track of output
test_outputs = np.zeros([len(norm_test), 2])
train_outputs = np.zeros([len(norm_train), 2])

clf = RandomForestClassifier(n_estimators=300,
	criterion='gini', oob_score=True, min_samples_split = 3, max_depth=12)
clf.fit(norm_train, train_out)

for i in range(len(norm_test)):
	test_outputs[i][0] = i
	test_outputs[i][1] = clf.predict(np.reshape(norm_test[i], (1,-1)))
for j in range(len(norm_train)):
	train_outputs[j][0] = j
	train_outputs[j][1] = clf.predict(np.reshape(norm_train[i], (1,-1)))

print('training score:', clf.score(norm_train, train_out))
print('validation score:', clf.score(norm_val, val_out))
print('out of bag:', clf.oob_score_)
np.savetxt("test_submission-10.csv", test_outputs, fmt = "%d", delimiter = ",")
np.savetxt("training_predict-1.csv", test_outputs, fmt = "%d", delimiter = ",")
