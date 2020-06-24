'''
@ author: Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

import os
import time
import pickle

import networkx as nx
from itertools import combinations
from scipy.stats.mstats import gmean
import numpy as np

import matplotlib.pyplot as plt
import scipy.io
import math
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score

import random
import tensorflow as tf


def save_the_file(file, output_path, file_name):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    f = open(output_path + "/" + file_name + ".p", "wb")
    pickle.dump(file, f)
    f.close()
    print("-------- Data Saved --------")

"""
case of unweighted
"""
def treat_Warning1(input):
    ret = []
    for row in input:
        new_row = []
        for i in row:
            new_row.append(i - 1)
        ret.append(tuple(new_row))
    return ret

"""
case of weighted
"""
def treat_Warning2(input):
    ret = []
    new_dict = dict()
    for row in input:
        for i in range(len(row)):
            temp = list(row)
            temp[i] -= 1
        if (tuple(temp) in new_dict.keys()):
            new_dict[tuple(temp)] += 1
        else:
            new_dict[tuple(temp)] = 1

    keys = new_dict.keys()
    for elem in keys:
        count = new_dict[elem]
        _elem = list(elem)
        _elem.append(count)
        ret.append(tuple(_elem))
    return ret 

class EWL:
    def __init__(self, graph, w_graph, configs):
        self.graph = graph
        self.w_graph = w_graph
        self.h_size = configs["h_size"]
        self.epochs = configs["epochs"]
        self.exist_node2vec = configs["exist_node2vec"]
        if self.exist_node2vec:
            #from node2vec.edges import HadamardEmbedder
            #from gensim.test.utils import common_texts, get_tmpfile
            from gensim.models import Word2Vec
            self.node2vec_model = Word2Vec.load("./n_weighted.model")
        else:
            self.node2vec_model = None

    def fit_classifier(self, train, labels):
        input_size = train.shape[1]
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=input_size))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(8, activation='relu'))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(train, labels, epochs=self.epochs)

    def find_subgraph_nodes(self, query):
        subgraph_nodes = set()
        for idx_node in query:
            length = nx.single_source_shortest_path_length(self.graph, idx_node,self.h_size)  # {5: 0, 0: 1, 6: 1, 10: 1, 16: 1, 1: 2, 2: 2, 3: 2, 4: 2, 7: 2, 8: 2, 11: 2, 12: 2, 13: 2, 17: 2, 19: 2, 21: 2, 31: 2}
            neighbors = list(length.keys())
            # neighbors.remove(idx_node) # remove itself
            subgraph_nodes = subgraph_nodes | set(neighbors)
        return list(subgraph_nodes)

    def extract_subgraph(self, query):
        subgraph_nodes = self.find_subgraph_nodes(query)
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        return subgraph

    def compute_geometric_mean(self, subgraph, query):
        tmp_subgraph = subgraph.copy()

        ''' Remove combination edges in query '''
        query_edges = list(combinations(query, 2))  # list of tuples
        for remove_edge in query_edges:
            try:
                tmp_subgraph.remove_edge(remove_edge[0], remove_edge[1])
            except:
                # query node 간에 직접적인 edge가 없는 경우
                pass

        ''' Construct reachable table '''
        query_reachable_table = dict()
        for query_node in query:
            query_reachable_table[query_node] = nx.descendants(tmp_subgraph, query_node)

        subgraph_nodes_not_query = set(tmp_subgraph.nodes) - set(query)
        n_nodes = tmp_subgraph.number_of_nodes()
        for idx_node in subgraph_nodes_not_query:
            shortest_path_lengths = []
            for query_node in query:
                if idx_node in query_reachable_table[query_node]:
                    if self.exist_node2vec:  ##### node2vec model을 사용
                        query_node_vector = self.node2vec_model.wv.get_vector(str(query_node))
                        idx_node_vector = self.node2vec_model.wv.get_vector(str(idx_node))
                        # shortest_path_lengths.append(np.linalg.norm(query_node_vector - idx_node_vector))
                        shortest_path_lengths.append(np.matmul(query_node_vector, idx_node_vector))
                    else:  ##### node2vec model을 사용하지 않는 경우
                        shortest_path_lengths.append(nx.shortest_path_length(tmp_subgraph, source=query_node, target=idx_node))


                else:
                    # WLNM에서는 이런 식으로 넣었는데 다른 방법??
                    shortest_path_lengths.append(n_nodes ** 2)
            mean = 0.0
            for elem in shortest_path_lengths:
                mean += elem
            mean /= len(shortest_path_lengths) 
            # tmp_subgraph.nodes[idx_node]['avg_dist'] = gmean(shortest_path_lengths)
            tmp_subgraph.nodes[idx_node]['avg_dist'] = mean

        ###### query edge들을 더해줘야하나???? node에 avg_dist 정도만 더해줘도 되지 않을까?
        for query_node in query:
            tmp_subgraph.nodes[query_node]['avg_dist'] = 0.0

        ########
        #print ("Finished - compute_gm ")
        return tmp_subgraph

    def sample(self, subgraph, nodelist, query):
        adj_matrix = np.asarray(nx.adj_matrix(subgraph, nodelist=nodelist).todense())
        vector = np.array([])
        for idx in range(len(adj_matrix)):  # 0, 1, 2
            if idx < len(query):
                vector = np.append(vector, adj_matrix[idx, len(query):])
            elif idx == len(adj_matrix) - 1:
                pass
            else:
                vector = np.append(vector, adj_matrix[idx, idx + 1:])
        #print ("Finished - sample ")
        return vector  ##### np.array

    def convert_into_adj_vector(self, query_data):
        #start = time.time()
        adj_vectors = None

        for idx in range(len(query_data)):
            query = query_data[idx]
            subgraph = self.extract_subgraph(query=query)
            subgraph = self.compute_geometric_mean(subgraph=subgraph, query=query)

            avg_dist = nx.get_node_attributes(subgraph, 'avg_dist')

            ### change-reverse order ###
            _nodelist = sorted(avg_dist.items(), key=(lambda x: x[1]))
            _nodelist.reverse()
            nodelist = list(map(lambda x: x[0], _nodelist))
            truncate_size = len( query) * 10 * self.h_size  ##### query가 커질 수록 subgraph가 커져야하니까 비례하도록 설정, 추후 zero padding을 어떻게 해야할 지 고민해야함.
            if len(nodelist) > truncate_size:
                nodelist = nodelist[:truncate_size]
            # len_node_lists.append(len(nodelist))
            adj_vector = self.sample(subgraph, nodelist, query)
            # print (adj_vector.shape)
            if idx == 0:
                adj_vectors = adj_vector
            else:

                try:
                    adj_vectors = np.vstack((adj_vectors, adj_vector))
                except:
                    # print(adj_vectors.shape, adj_vector.shape)
                    ''' matching dimension and zero-padding '''
                    # print ("----------",,'-----------')
                    # print (adj_vectors.shape, adj_vector.shape)
                    if idx == 1:  ##### 이렇게 idx ==1 해도 문제가 Train 할 때 문제가 없는가?
                        dimension1 = adj_vectors.shape[0]
                    else:
                        dimension1 = adj_vectors.shape[1]

                    dimension2 = adj_vector.shape[0]
                    if dimension1 > dimension2:
                        adj_vector = np.append(adj_vector, np.zeros(dimension1 - dimension2))
                        adj_vectors = np.vstack((adj_vectors, adj_vector))
                    elif dimension1 < dimension2:
                        if idx == 1:
                            adj_vectors = np.hstack((adj_vectors, np.zeros(dimension2 - dimension1)))
                        else:
                            adj_vectors = np.hstack(
                                (adj_vectors, np.zeros((adj_vectors.shape[0], dimension2 - dimension1))))
                        adj_vectors = np.vstack((adj_vectors, adj_vector))
                    else:
                        # This episode will catch by try statement.
                        pass
            # print("adj_vector", adj_vector[:10])
            # print (len(adj_vectors), type(adj_vector))
        # adj_vectors = np.asarray(adj_vectors)
        # print (adj_vectors.shape)
        return adj_vectors

    def train(self, query_train, query_labels):
        train_adj_vectors = self.convert_into_adj_vector(query_train)

        print("train adj_vectors shape : ", train_adj_vectors.shape)

        # self.fit_classifier(train_adj_vectors, query_labels)

        return train_adj_vectors

    def predict(self, query_test, query_labels):
        test_adj_vectors = self.convert_into_adj_vector(query_test)
        # if is_save:
        #    save_the_file(test_adj_vectors, "./data", "test_adj_vectors")
        print("Test adj_vectors shape : ", test_adj_vectors.shape)

        # test_adj_vectors_emb = TSNE(n_components=189).fit_transform(test_adj_vectors)
        # predictions = self.model.predict(test_adj_vectors_emb)
        # predictions = self.model.predict(test_adj_vectors)
        test_loss, test_acc = self.model.evaluate(test_adj_vectors, query_labels, verbose=2)
        # predictions = self.model.predict(test_adj_vectors)
        print(test_loss, test_acc)
        return test_adj_vectors, test_acc


if __name__ == "__main__":
    ########## Hyper parameter ##########

    is_save = True

    configs = {
        "graph": "paper_author edge version",
        "pos communities": "paper_author raw",
        "neg communities": "negative sampling",
        "Test": "query public raw",
        "sample": 0.8,
        "h_size": 2,
        "epochs": 50,
        "exist_node2vec": True,
        "batch_size" : 1000,
        "weighted": False,
        "datasetPath": "./w_1000_dataset" # path to save datasets for training
    }

    ########## Bulid the Graph ##########

    weighted = configs["weighted"]
    if (weighted):
        inputDir = "./w_input"
    else:
        inputDir = "./matmul_input/"
    if not os.path.exists(inputDir): 
        os.mkdir(inputDir)

    datasetPath = configs["datasetPath"]
    if not os.path.exists(datasetPath):
        os.mkdir(datasetPath)

    if (weighted):
        wf = open("./pa_edges_list.p", "rb")  # set of tuple
        w_edges = pickle.load(wf)
        wf.close()
    f = open("./pa_edges_set.p", "rb")
    pa_edges = pickle.load(f)
    f.close()

    n_nodes = max(list(map(lambda x: max(x), pa_edges)))
    n_links = len(pa_edges)
    print("# of total nodes : ", n_nodes)
    print("# of total links : ", n_links)

    if (weighted):
        w_graph_edges = treat_Warning2(w_edges)
        graph_edges = treat_Warning1(pa_edges)
    else:
        graph_edges = treat_Warning1(pa_edges)

    network = nx.empty_graph()
    w_network = nx.empty_graph()
    if (weighted):
        w_network.add_weighted_edges_from(w_graph_edges)
        network.add_edges_from(graph_edges)
    else:
        network.add_edges_from(graph_edges)

    for node_idx in range(n_nodes):
        if node_idx not in list(network.nodes):
            network.add_node(node_idx)

    if (weighted):
        for node_idx in range(n_nodes):
            if node_idx not in list(w_network.nodes):
                w_network.add_node(node_idx)


    ''' Train '''
    f = open(os.path.join(inputDir, "pos_communities.p"), "rb") # new?
    _pos_communities = pickle.load(f)
    f.close()

    pos_communities = _pos_communities["data"]
    pos_communities = treat_Warning1(pos_communities)

    f = open(os.path.join(inputDir, "neg_communities.p"), "rb") # new!
    _neg_communities = pickle.load(f)
    f.close()

    neg_communities = _neg_communities["data"]

    query_Train_edges = pos_communities + neg_communities
    query_Train_labels = [1] * len(pos_communities) + [0] * len(neg_communities)

    query_Train_edges, query_Train_labels = shuffle(query_Train_edges, query_Train_labels)

    print ("# of Train edges : ", len(query_Train_edges))

    ''' Test '''
    f = open(os.path.join(inputDir, "qpu_edges_raw_duplicate.p"), 'rb') 
    query_Test_edges = pickle.load(f)
    f.close()
    query_Test_edges = treat_Warning1(query_Test_edges)

    f = open(os.path.join(inputDir, "apu_labels_raw_duplicate.p"), 'rb') 
    query_Test_labels = pickle.load(f)
    f.close()

    print("# of Test edges : ", len(query_Test_edges))

    ############################################################################################

    f = open(os.path.join(inputDir, "configs.txt"), "w")
    f.write(str(configs))
    f.close()


    ewl = EWL(graph=network, w_graph=w_network, configs=configs)

    import multiprocessing as mp

    num_train_batch = int(len(query_Train_edges)/configs["batch_size"])
    num_test_batch = int(len(query_Test_edges) / configs["batch_size"])
    print("num_train_batch: {}, num_test_batch: {}".format(num_train_batch, num_test_batch))

    start = time.time()

    f = open(os.path.join(inputDir, "query_Train_labels.p"), "wb")
    pickle.dump(query_Train_labels, f)
    f.close()

    f = open(os.path.join(inputDir, "query_Test_labels.p"), "wb")
    pickle.dump(query_Test_labels, f)
    f.close()
    print("query_Train_labels: {}, query_Test_labels: {}".format(len(query_Train_labels), len(query_Test_labels)))

    for idx_batch in range(num_train_batch):
        print ("=======", idx_batch, "=======")
        if idx_batch != num_train_batch - 1:
            train_adj_vectors = ewl.train(query_train=query_Train_edges[idx_batch*configs["batch_size"]: (idx_batch+1)*configs["batch_size"]], query_labels=query_Train_labels[idx_batch*configs["batch_size"]: (idx_batch+1)*configs["batch_size"]])
        else:
            train_adj_vectors = ewl.train(query_train=query_Train_edges[idx_batch * configs["batch_size"]:], query_labels=query_Train_labels[idx_batch * configs["batch_size"]:])
        print (time.time() - start)


        f = open(os.path.join(datasetPath, "train_adj_vectors_"+str(idx_batch)+".p"), "wb")
        pickle.dump(train_adj_vectors,f)
        f.close()

    for idx_batch in range(num_test_batch):
        print ("=======", idx_batch, "=======")
        if idx_batch != num_test_batch - 1:
            test_adj_vectors = ewl.train(query_train=query_Test_edges[idx_batch*configs["batch_size"]: (idx_batch+1)*configs["batch_size"]], query_labels=query_Test_labels[idx_batch*configs["batch_size"]: (idx_batch+1)*configs["batch_size"]])
        else:
            test_adj_vectors = ewl.train(query_train=query_Test_edges[idx_batch * configs["batch_size"]:], query_labels=query_Test_labels[idx_batch * configs["batch_size"]:])
        print (time.time() - start)


        f = open(os.path.join(datasetPath, "test_adj_vectors_"+str(idx_batch)+".p"), "wb")
        pickle.dump(test_adj_vectors,f)
        f.close()
