'''
@ author: Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

import os
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def training(configs):

    start = time.time()

    total_epochs = configs["total_epochs"]
    type_n2v = configs["type_n2v"]
    data_type = configs["data_type"]
    dirname = configs["basedir"]+"encoded_vector/" + type_n2v + "/"+str(configs["h_size"])+"/"+data_type + "/"
    outdirname = "../output/" + type_n2v + "/"+str(configs["h_size"])+"/"+data_type + "/"
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)
    n_hidden = configs["n_hidden"]
    batch_size = configs["batch_size"]

    f = open(outdirname + "configs.txt", "w")
    for k in configs.keys():
        f.write(str(k) + " : " + str(configs[k]))
        f.write("\n")
    f.close()

    n_Train_files = 0
    n_Test_files = 0

    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        if "train_adj_vectors" in full_filename:
            n_Train_files += 1
        if "test_adj_vectors" in full_filename:
            n_Test_files += 1


    input_size= 0
    for idx_file in range(n_Train_files):
        print ("Training - Finding input size...", idx_file)
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
            batch_vectors = np.load(dirname + "train_adj_vectors_" + str(idx_file)+".npy")
            if batch_vectors.shape[1] > input_size:
                input_size = batch_vectors.shape[1]


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden[0], activation='relu', input_dim=input_size))
    model.add(tf.keras.layers.Dense(n_hidden[1], activation='relu'))
    model.add(tf.keras.layers.Dense(n_hidden[2], activation='relu'))
    model.add(tf.keras.layers.Dense(n_hidden[3], activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    epoch_loss_list = []
    epoch_acc_list = []

    for idx_epoch in range(total_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0

        Train_data = "None"
        Label_data = []
        count = 1

        for idx_file in range(n_Train_files):
            print("Traing - Epoch : ", idx_epoch, "File idx : ", idx_file)
            if data_type == "pickle":
                f = open(dirname + "train_adj_vectors_" + str(idx_file) + ".p", 'rb')
                batch_dict = pickle.load(f)
                f.close()

                for query in list(batch_dict.keys()):
                    one_adj_vector = batch_dict[query][1]
                    if one_adj_vector.shape[0] < input_size:
                        one_adj_vector = np.hstack( (one_adj_vector, np.zeros(input_size - one_adj_vector.shape[0])))
                    if Train_data == "None":
                        Train_data = one_adj_vector
                    else:
                        Train_data = np.vstack((Train_data, one_adj_vector))
                    Label_data.append(batch_dict[query][2])
                    if Train_data.shape[0] == batch_size:
                        hist = model.fit(Train_data, Label_data, epochs=1)
                        epoch_loss += hist.history['loss'][0]
                        epoch_acc += hist.history["acc"][0]
                        Train_data = "None"
                        Label_data = []
                        count += 1
                    else:
                        pass
                if idx_file == n_Train_files - 1 and Train_data != "None":
                    hist = model.fit(Train_data, Label_data, epochs=1)
                    epoch_loss += hist.history['loss'][0]
                    epoch_acc += hist.history["acc"][0]
                    Train_data = "None"
                    Label_data = []
                    count += 1
            elif data_type == "npy":
                batch_vectors = np.load(dirname + "train_adj_vectors_" + str(idx_file) + ".npy")
                batch_labels = np.load(dirname + "train_adj_labels_" + str(idx_file) + ".npy")
                hist = model.fit(batch_vectors, batch_labels, epochs=1)
                epoch_loss += hist.history['loss'][0]
                epoch_acc += hist.history["acc"][0]
                count += 1
        epoch_loss_list.append(epoch_loss / count)
        epoch_acc_list.append(epoch_acc / count)

    model.save(outdirname + 'model.h5')

    fig, (ax1, ax2)= plt.subplots(1,2)
    ax1.plot(range(total_epochs), epoch_loss_list)
    ax1.set_xlabel("# of Epoch")
    ax1.set_ylabel("Training loss")
    ax2.plot(range(total_epochs), epoch_acc_list)
    ax2.set_xlabel("# of Epoch")
    ax2.set_ylabel("Training accuracy")
    plt.savefig(outdirname + "Train_result_epoch" + str(total_epochs) + ".png")

    print("Training - Elapsed time : ", time.time() - start)