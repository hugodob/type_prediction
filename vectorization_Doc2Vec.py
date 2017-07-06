import numpy as np
import matplotlib.pyplot as plt
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
import re

#Lowers the name and put it to the LabeledSentence format
def format_name(name):
    name=name.lower()
    name= re.sub(r'[^\w\s]','',name)
    return name

def format_data(data):
    for i in range(len(data[:][0])):
        data[i][0]=format_name(data[i][0])
    #Important to shuffle now if we want to have consistent results
    np.random.shuffle(data)
    return data

#Formats all the data to LabeledSentence as needed for Doc2Vec model's inputs
def format_labeled_sentences(data):
    sentences=[]
    for i in range(len(data)):
        sentences.append(LabeledSentence(data[i].split(),[str(i)]))
    return sentences

#This function builds the model, trains it and saves it in a file indicating the chosen parameters
def train_Doc2Vec(sentences, min_count, window, size, sample, negative, workers, nb_epochs):
    model=Doc2Vec(min_count=min_count, window=window, size=size, sample=sample, negative=negative, workers=workers)
    model.build_vocab(sentences)
    print('start training')
    for epoch in range(nb_epochs):
        print('starting %d epoch'%(epoch))
        #Important to shuffle between each epoch if we want a good generalization
        shuffle(sentences)
        model.train(sentences, total_examples=46387, epochs=nb_epochs)
    model.save('/Doc2Vec models/imdb_%d_%d_%d_%f_%d_%d_%d' %(min_count, window, size, sample, negative, workers, nb_epochs))
    return model
