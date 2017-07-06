from DataSet import DataSet
import classification
import vectorization_Doc2Vec
from gensim.models import Doc2Vec
import numpy as np

def main():
    data_set= DataSet(None, None)
    data_set.import_from_csv('Data set/field_hugo.csv')
    fields=["displayName", "type"]
    data_set.extract_fields_data(fields)
    list_labels=['STRING', 'TEXT', 'PERSON', 'DATE', 'INTEGER', 'BOOLEAN', 'DECIMAL', 'DATETIME', 'URL']
    data_set.clear_data_in_list("type", list_labels)
    data=vectorization_Doc2Vec.format_data(data_set.data)
    np.savetxt('Data set/test.out', data_set.data[:,1], fmt='%s')
    sentences= vectorization_Doc2Vec.format_labeled_sentences(data[:,0])
    model=vectorization_Doc2Vec.train_Doc2Vec(sentences, 1, 10, 40, 1e-4, 5, 8, 10)
    #model=Doc2Vec.load('Doc2Vec models/imdb_1_10_%d_0.000100_5_8_10' %(size_vectors))
    labels=np.loadtxt("Data set/test.out", dtype=str)
    train_arrays, train_labels, test_arrays, test_labels=classification.prepare_data_set(labels, list_labels, model, 40)
    mlp=classification.train_class(40, train_arrays, train_labels, test_arrays, test_labels)
    testscore=mlp.evaluate(test_arrays,test_labels,verbose=1)
    print("test score")
    print(testscore)
    return
main()
