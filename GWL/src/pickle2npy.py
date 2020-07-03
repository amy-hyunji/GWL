'''
@ author: Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

import pickle
import numpy as np
import time
import os

def pickle2npy(configs):
    '''
    When user want spend time, it can be solution that load the npy data instead of load pickle
    '''
    start = time.time()
    if configs["exist_node2vec"]:
        type_n2v = "n2v"
    else:
        type_n2v = "nn2v"
    dirname = configs["basedir"]+"encoded_vector/" + type_n2v + "/"+str(configs["h_size"])+"/pickle/"
    outdirname = configs["basedir"]+"encoded_vector/" + type_n2v + "/"+str(configs["h_size"])+"/npy/"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)

    n_Train_files = 0
    n_Test_files = 0

    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        if "train_adj_vectors" in full_filename:
            n_Train_files += 1
        if "test_adj_vectors" in full_filename:
            n_Test_files += 1

    input_size = 0
    for idx_file in range(n_Train_files):
        f = open(dirname + "train_adj_vectors_" + str(idx_file) + ".p", 'rb')
        batch_dict = pickle.load(f)
        f.close()
        for query in list(batch_dict.keys()):
            if batch_dict[query][1].shape[0] >= input_size:
                input_size = batch_dict[query][1].shape[0]
            else:
                pass

    for idx1 in range(n_Train_files):
        print ("pickle2npy - Training", idx1)
        f = open(dirname + "train_adj_vectors_" + str(idx1) + ".p", "rb")
        batch_dict = pickle.load(f)
        f.close()
        batch_vectors = "None"
        batch_labels = []
        for idx2 in list(batch_dict.keys()):
            one_adj_vector = batch_dict[idx2][1]
            if one_adj_vector.shape[0] < input_size:
                one_adj_vector = np.hstack((one_adj_vector, np.zeros(input_size - one_adj_vector.shape[0])))
            if batch_vectors == "None":
                batch_vectors = one_adj_vector
            else:
                batch_vectors = np.vstack((batch_vectors, one_adj_vector))
            batch_labels.append(batch_dict[idx2][2])
        np.save(outdirname+"train_adj_vectors_" + str(idx1), batch_vectors)
        np.save(outdirname+"train_adj_labels_" + str(idx1), np.array(batch_labels))

    for idx1 in range(n_Test_files):
        print ("pickle2npy - Testing", idx1)
        f = open(dirname + "test_adj_vectors_" + str(idx1) + ".p", "rb")
        batch_dict = pickle.load(f)
        f.close()
        batch_vectors = "None"
        batch_labels = []

        for idx2 in list(batch_dict.keys()):
            one_adj_vector = batch_dict[idx2][1]
            if one_adj_vector.shape[0] < input_size:
                one_adj_vector = np.hstack((one_adj_vector, np.zeros(input_size - one_adj_vector.shape[0])))
            if batch_vectors == "None":
                batch_vectors = one_adj_vector
            else:
                batch_vectors = np.vstack((batch_vectors, one_adj_vector))
            batch_labels.append(batch_dict[idx2][2])
        np.save(outdirname+"test_adj_vectors_" + str(idx1), batch_vectors)
        np.save(outdirname+"test_adj_labels_" + str(idx1), np.array(batch_labels))
    print ("pickle2npy - Elapsed time : ", time.time() - start)