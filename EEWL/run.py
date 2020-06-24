import os
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

total_epochs = 10
type_n2v = "n2v" ##### n2v(use node2vec) or nn2v (not use node2vec)
dirname = "./input/input_batch/"+type_n2v+"/"
input_size = 30825 ##### 휴리스틱

n_Train_files = 0
n_Test_files = 0

filenames = os.listdir(dirname)
for filename in filenames:
    full_filename = os.path.join(dirname, filename)
    if "train_adj_vectors" in full_filename:
        n_Train_files += 1
    if "test_adj_vectors" in full_filename:
        n_Test_files += 1

###############################################################
f = open(dirname+"query_Train_labels.p", "rb")
total_train_label = pickle.load(f)
f.close()
###############################################################
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=input_size))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
##################################################################
epoch_loss_list = []
epoch_acc_list = []

for idx_epoch in range(total_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0

    for idx_file in range(n_Train_files):
        print("=====", idx_epoch, idx_file, "=====")
        f = open(dirname+"train_adj_vectors_" + str(idx_file) + ".p", 'rb')
        batch_data = pickle.load(f)
        f.close()

        if batch_data.shape[1] < input_size:
            batch_data = np.hstack((batch_data, np.zeros((batch_data.shape[0], input_size - batch_data.shape[1]))))

        if idx_file != n_Train_files - 1:
            label = total_train_label[idx_file * 1000: (idx_file + 1) * 1000]
        else:
            label = total_train_label[idx_file * 1000:] ##### 마지막은 imbalance된 batch
        hist = model.fit(batch_data, label, epochs=1)
        epoch_loss += hist.history['loss'][0]
        epoch_acc += hist.history["acc"][0]
    epoch_loss_list.append(epoch_loss/n_Train_files)
    epoch_acc_list.append(epoch_acc / n_Train_files)

print ("End Training")

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(range(total_epochs), epoch_loss_list, marker="O")
ax2.plot(range(total_epochs), epoch_acc_list, marker="O")

plt.savefig("Train_result.png")

##################################################################
print ("Start Testing")
f = open(dirname+"query_Test_labels.p", "rb")
total_test_label = pickle.load(f)
f.close()

lss = 0.0
acc = 0.0

for idx in range(n_Test_files):
    print("=====", idx, "=====")
    f = open(dirname+"test_adj_vectors_" + str(idx) + ".p", 'rb')
    batch_data = pickle.load(f)
    f.close()

    if batch_data.shape[1] < input_size:
        batch_data = np.hstack((batch_data, np.zeros((batch_data.shape[0], input_size - batch_data.shape[1]))))

    if idx != n_Test_files - 1:
        label = total_test_label[idx * 1000: (idx + 1) * 1000]
    else:
        label = total_test_label[idx * 1000:]
    batch_loss, batch_acc = model.evaluate(batch_data, label, verbose=2)
    lss += batch_loss
    acc += batch_acc
print ("Average test loss : ", lss / n_Test_files)
print ("Average test accuracy : ", acc / n_Test_files)