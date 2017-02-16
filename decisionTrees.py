import numpy as np 
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


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
# load data from CSV files
# extract data into data_array by row, first row is column labels
with open('train_2008.csv', 'r') as srcfile:
	data_iter = csv.reader(srcfile, quotechar = '"')
	data = [data for data in data_iter]
	data = data[1:]
	data = np.asarray(data)
	np.random.shuffle(data)

X_train = (np.asarray(data))[:, 3:382].astype(np.float)
y_train = (np.asarray(data))[:, 382].astype(np.float)

with open('test_2008.csv', 'r') as srcfile:
	data_iter = csv.reader(srcfile, quotechar = '"')
	data = [data for data in data_iter]
	data = data[1:]

X_test = (np.asarray(data))[:, 3:382].astype(np.float)
# split the training data into a training and a validation set
val_set = X_train[0:1000,]
val_out = y_train[0:1000]
train_set = X_train[1000:len(X_train), ]
train_out = y_train[1000:len(X_train)]
norm_packet = normalize(train_set)

# normalize the training data; normalize validation data with the same mean 
# and standard deviation used for training data
norm_train = norm_packet[1]
scaler = norm_packet[0]
norm_val = scaler.transform(val_set)

min_split_sizes = range(1700, 1701, 1)
validation_errors = np.zeros(len(min_split_sizes))
training_errors = np.zeros(len(min_split_sizes))

# normalize test input
norm_test = scaler.transform(X_test)

# keep track of output
test_outputs = np.zeros([len(norm_test), 2])

for i in range(len(min_split_sizes)):
	split_size = min_split_sizes[i]
	print "Fitting decision tree with split size ", split_size
	# train the decision tree classifier with the give min_split_size
	clf = DecisionTreeClassifier(min_samples_split = split_size, criterion = "entropy")
	clf.fit(norm_train, train_out)

	# get the classification training and validation error of this classifier
	training_errors[i] = error(clf, norm_train, train_out)
	validation_errors[i] = error(clf, norm_val, val_out)

	for i in range(len(norm_test)):
		test_outputs[i][0] = i
		test_outputs[i][1] = clf.predict(np.reshape(norm_test[i], (1,-1)))

print "E_in: ", training_errors[0]
print "E_val: ", validation_errors[0] 
np.savetxt("test_submission.csv", test_outputs, fmt = "%d", delimiter = ",")

# plot the training and validation errors as a function of min_split_size 
#plt.plot(min_split_sizes, training_errors, 'b')
#plt.plot(min_split_sizes, validation_errors, 'r')
#plt.title("Minimum Split Size vs. Error With Entropy")
#plt.show()

