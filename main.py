from DataSet import DataSet
import classification
import vectorization_Doc2Vec
from gensim.models import Doc2Vec
import numpy as np

def main():
    data_set= DataSet(None, None)
    data_set.import_from_csv('field_hugo.csv')
    fields=["displayName", "type"]
    data_set.extract_fields_data(fields)
    list_labels=['STRING', 'TEXT', 'PERSON', 'DATE', 'INTEGER', 'BOOLEAN', 'DECIMAL', 'DATETIME', 'URL']
    data_set.clear_data_in_list("type", list_labels)
    data=vectorization_Doc2Vec.format_data(data_set.data)
    np.savetxt('test.out', data_set.data[:,1], fmt='%s')
    size=[20, 40, 60, 80]
    sentences= vectorization_Doc2Vec.format_labeled_sentences(data[:,0])
    for  size_vectors in size:
        model=vectorization_Doc2Vec.train_Doc2Vec(sentences, 1, 10, size_vectors, 1e-4, 5, 8, 10)
        list_labels=['STRING', 'TEXT', 'PERSON', 'DATE', 'INTEGER', 'BOOLEAN', 'DECIMAL', 'DATETIME', 'URL']
        #model=Doc2Vec.load('./imdb_1_10_%d_0.000100_5_8_10' %(size_vectors))
        labels=np.loadtxt("test.out", dtype=str)
        train_arrays, train_labels, test_arrays, test_labels=classification.prepare_data_set(labels, list_labels, model, size_vectors)
        classification.train_class(size_vectors, train_arrays, train_labels, test_arrays, test_labels)
    return
main()
