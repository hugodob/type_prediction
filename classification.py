import numpy as np
from gensim.models import Doc2Vec
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation, Dropout


#Fits the data set in a training set (80% of the data) and a test set (20% of the data)
def prepare_data_set(labels, list_labels, model, size_vectors):
    n_class=len(list_labels)
    n=len(labels)
    n_train=int(0.8*n)
    n_test=n-n_train
    train_arrays=np.zeros((n_train, size_vectors))
    train_labels=np.zeros((n_train,n_class))
    test_arrays=np.zeros((n_test, size_vectors))
    test_labels=np.zeros((n_test,n_class))
    for i in range(n_train):
        train_arrays[i]=model.docvecs[str(i)]
        #Here the labels are converted to 9 elements vectors, each vector corresponding to a certain type
        train_labels[i][list_labels.index(labels[i])]=1
    for i in range(n_test):
        test_arrays[i]=model.docvecs[str(i+n_train)]
        test_labels[i][list_labels.index(labels[i+n_train])]=1
    return train_arrays, train_labels, test_arrays, test_labels


#This function builds the MLP, trains it and saves it
def train_class(size_vectors, train_arrays, train_labels, test_arrays, test_labels):
    mlp = Sequential()
    mlp.add(Dense(500,input_dim=size_vectors,activation='relu', kernel_initializer='lecun_uniform'))
    mlp.add(Dropout(0.2))
    mlp.add((Dense(500,activation='relu', kernel_initializer='lecun_uniform')))
    mlp.add(Dropout(0.2))
    mlp.add((Dense(50,activation='relu', kernel_initializer='lecun_uniform')))
    mlp.add(Dropout(0.2))
    mlp.add((Dense(9,activation="softmax", kernel_initializer='lecun_uniform')))
	# Compile model
    mlp.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    mlp.fit(train_arrays,train_labels,epochs=50,batch_size=50, validation_split=.05)
    mlp.save('classification models/my_model.h5')
    return mlp
