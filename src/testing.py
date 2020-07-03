'''
@ author: Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

import os
import time
import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

def testing(configs):
    start = time.time()

    type_n2v = configs["type_n2v"]
    data_type = configs["data_type"]
    dirname = configs["basedir"]+"encoded_vector/" + type_n2v + "/" + str(configs["h_size"]) + "/" + data_type + "/"
    outdirname = "../output/" + type_n2v + "/" + str(configs["h_size"]) + "/" + data_type + "/"

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
        print ("Testing - Finding input size...", idx_file)
        if data_type == "pickle":
            f = open(dirname + "train_adj_vectors_" + str(idx_file) + ".p", 'rb')
            batch_dict = pickle.load(f)
            f.close()
            for query in list(batch_dict.keys()):
                if batch_dict[query][1].shape[0] >= input_size:
                    input_size = batch_dict[query][1].shape[0]
                else:
                    pass
        elif data_type == "npy":
            batch_vectors = np.load(dirname + "train_adj_vectors_" + str(idx_file) + ".npy")
            if batch_vectors.shape[1] > input_size:
                input_size = batch_vectors.shape[1]
    model = tf.keras.models.load_model(outdirname+'model.h5')
    print (model.summary())


    lss = []
    acc = []

    predictions = []
    total_labels = []

    for idx in range(n_Test_files):
        print("Testing - File idx : ", idx)
        if data_type == "pickle":
            f = open(dirname + "test_adj_vectors_" + str(idx) + ".p", 'rb')
            batch_dict = pickle.load(f)
            f.close()

            for query in list(batch_dict.keys()):
                one_adj_vector = batch_dict[query][1]
                if one_adj_vector.shape[0] < input_size:
                    one_adj_vector = np.hstack((one_adj_vector, np.zeros(input_size - one_adj_vector.shape[0])))
                batch_loss, batch_acc = model.evaluate(one_adj_vector, batch_dict[query][2], verbose=2)
                lss.append(batch_loss)
                acc.append(batch_acc)
        elif data_type == "npy":
            batch_vectors = np.load(dirname + "test_adj_vectors_" + str(idx) + ".npy")
            batch_labels = np.load(dirname + "test_adj_labels_" + str(idx) + ".npy")
            batch_loss, batch_acc = model.evaluate(batch_vectors, batch_labels, verbose=2)
            batch_pred = model.predict(batch_vectors)
            lss.append(batch_loss)
            acc.append(batch_acc)
            for row_idx in range(batch_pred.shape[0]):
                row = batch_pred[row_idx]
                if row[1] >= row[0]:
                    predictions.append(1)
                else:
                    predictions.append(0)
                total_labels.append(batch_labels[row_idx])

    loss = sum(lss) / len(lss)
    accuracy = sum(acc) / len(acc)

    print("Testing - Average test loss : ", loss)
    print("Testing - Average test accuracy : ", accuracy)

    confusion_mat = confusion_matrix(total_labels, predictions)
    print("Testing - Confusion matrix : ", confusion_mat)

    f = open(outdirname + "result.txt", "w")
    for k in configs.keys():
        f.write(str(k) + " : " + str(configs[k]))
        f.write("\n")
    f.write("Test loss : " + str(loss))
    f.write("\n")
    f.write("Test accuracy : " + str(accuracy))
    f.write("\n")
    f.write("Confusion matrix : " + str(confusion_mat))
    f.close()
    print("Testing - Elapsed time : ", time.time() - start)
