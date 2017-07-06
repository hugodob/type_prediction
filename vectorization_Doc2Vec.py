import numpy as np
import matplotlib.pyplot as plt
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
import re

def format_name(name):
    name=name.lower()
    name= re.sub(r'[^\w\s]','',name)
    return name

def format_data(data):
    for i in range(len(data[:][0])):
        data[i][0]=format_name(data[i][0])
    np.random.shuffle(data)
    return data

def format_labeled_sentences(data):
    sentences=[]
    for i in range(len(data)):
        sentences.append(LabeledSentence(data[i].split(),[str(i)]))
    return sentences

def train_Doc2Vec(sentences, min_count, window, size, sample, negative, workers, nb_epochs):
    model=Doc2Vec(min_count=min_count, window=window, size=size, sample=sample, negative=negative, workers=workers)
    model.build_vocab(sentences)
    print('start training')
    for epoch in range(nb_epochs):
        print('starting %d epoch'%(epoch))
        shuffle(sentences)
        model.train(sentences, total_examples=46387, epochs=nb_epochs)
    model.save('/Doc2Vec models/imdb_%d_%d_%d_%f_%d_%d_%d' %(min_count, window, size, sample, negative, workers, nb_epochs))
    return model
