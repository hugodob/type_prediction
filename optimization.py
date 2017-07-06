#to be done
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
def create_model(size_vectors=40,nb_hidden_layers=1, kernel_initializer="lecun_uniform", neurons, dropout_rate=0.0, weight_constraint=0, loss='categorical_crossentropy', optimizer="Nadam", activation='relu', output_activation='softmax'):
	# create model
	model = Sequential()
	model.add(Dense(neurons[0], input_dim=size_vectors, kernel_initializer=kernel_initializer, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
    for i in range(nb_hidden_layers):
        model.add(Dense(neurons[1+i], kernel_initializer=kernel_initializer, activation=activation, kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
	model.add(Dense(9, kernel_initializer=kernel_initializer, activation=activation))
	# Compile model
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
	return model

def find_best_parameters()
