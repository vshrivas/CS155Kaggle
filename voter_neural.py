import csv
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1l2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# load data from CSV files
# extract data into data_array by row, first row is column labels
with open('train_2008.csv', 'r') as srcfile:
    data_iter = csv.reader(srcfile, quotechar = '"')
    data = [data for data in data_iter]
    data = data[1:]
    print(data)

def normalize(inputs):
	print ("Normalizing inputs...")
	scaler = StandardScaler().fit(inputs)
	norm_inputs = scaler.transform(inputs)
	print ("Done normalizing. ")
	return (scaler, norm_inputs)

# note: two columns removed for x
X_train = (np.asarray(data))[:, 3:382].astype(np.float)
y_train = (np.asarray(data))[:, 382].astype(np.float)

# print(np.shape(X_train))
# print(y_train)

norm_packet = normalize(X_train)
norm_train = norm_packet[1]
scaler = norm_packet[0]
norm_train = np.array(norm_train)

## In your homework you should transform each input data point
## into a single vector here and should transform the
## labels into a one hot vector using np_utils.to_categorical

# our results fall into two categories
y_train_hot = np.empty([0, 2])
X_train_hot = np.array(X_train)

y_train_hot -= 1

for i in range(0, y_train.size):
    y_train_hot = np.vstack((y_train_hot, to_categorical(y_train[i], 2)))

# create our model
model = Sequential()
model.add(Dense(1000, input_shape=(382,), init='normal', activation='sigmoid'))
model.add(Dense(1000, init='normal', activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1000, init='normal', activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1000, init='normal', activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
fit = model.fit(X_train_hot, y_train_hot, batch_size=1000, nb_epoch=30,verbose=1)
score = model.evaluate(X_train_hot, y_train_hot, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
