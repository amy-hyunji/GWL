'''
@ author: Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

import pickle
from GWL import GWL
import networkx as nx
from sklearn.utils import shuffle
import os
import time

def treat_Warning1(input):
    '''
    - if smallest node index is 1, created 0 node when creating networkx graph
    '''
    ret = []
    for row in input:
        new_row = []
        for i in row:
            new_row.append(i - 1)
        ret.append(tuple(new_row))
    return ret

def encoding(configs):
    ''' encode the subgraph to flat vector which is input of classifier '''

    ''' Bulid the Graph '''

    f = open(configs["basedir"]+"pa_edges_set.p", "rb")
    pa_edges_set = pickle.load(f)
    f.close()

    n_nodes = max(list(map(lambda x: max(x), pa_edges_set)))
    n_links = len(pa_edges_set)
    print("Encoding - # of total nodes in graph : ", n_nodes)
    print("Encoding - # of total links in graph : ", n_links)

    graph_edges = treat_Warning1(pa_edges_set)

    network = nx.empty_graph()
    network.add_edges_from(graph_edges)

    for node_idx in range(n_nodes):
        if node_idx not in list(network.nodes):
            network.add_node(node_idx)

    ''' Load Train data '''

    f = open(configs["basedir"]+configs["target"]+"/pos_communities.p", "rb")
    _pos_communities = pickle.load(f)
    f.close()
    pos_communities = _pos_communities["data"]
    pos_communities = treat_Warning1(pos_communities)

    f = open(configs["basedir"]+configs["target"]+"/neg_communities.p", "rb")
    _neg_communities = pickle.load(f)
    f.close()
    neg_communities = _neg_communities["data"]

    query_Train_edges = pos_communities + neg_communities
    query_Train_labels = [1] * len(pos_communities) + [0] * len(neg_communities)
    query_Train_edges, query_Train_labels = shuffle(query_Train_edges, query_Train_labels)

    print("Encoding - # of Train edges : ", len(query_Train_edges))

    ''' Load Test data '''

    f = open(configs["basedir"]+"qpu_edges_raw_duplicate.p", 'rb')
    _query_Test_edges = pickle.load(f)
    f.close()
    _query_Test_edges = treat_Warning1(_query_Test_edges)

    f = open(configs["basedir"]+"apu_labels_raw_duplicate.p", 'rb')
    _query_Test_labels = pickle.load(f)
    f.close()

    if configs["target"] == "edges":
        query_Test_edges = []
        query_Test_labels = []
        for idx in range(len(_query_Test_edges)):
            if len(_query_Test_edges[idx]) == 2:
                query_Test_edges.append(_query_Test_edges[idx])
                query_Test_labels.append(_query_Test_labels[idx])
    elif configs["target"] == "raw":
        query_Test_edges = _query_Test_edges
        query_Test_labels = _query_Test_labels
    else:
        print ("!!!Please check configs!!!")
        return
    print("Encoding - # of Test edges : ", len(query_Test_edges))

    ''' Encoding '''

    if configs["exist_node2vec"]:
        type_n2v = "n2v"
    else:
        type_n2v = "nn2v"

    dirname = configs["basedir"]+"encoded_vector/" + type_n2v + "/"+str(configs["h_size"])+"/pickle/"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    f = open(dirname + "configs.txt", "w")
    for k in configs.keys():
        f.write(str(k)+" : "+str(configs[k]))
        f.write("\n")
    f.close()

    gwl = GWL(graph=network, configs=configs)

    start = time.time()

    num_train_batch = int(len(query_Train_edges) / configs["batch_size"])

    for idx1 in range(num_train_batch):
        batch_dict = dict()
        cnt = 0
        for idx2 in range(configs["batch_size"]):
            if idx2 % 100 == 0:
                print("Encoding Train data - ", idx1, idx2)
            try:
                query = tuple(query_Train_edges[idx1 * configs["batch_size"] + idx2])
            except:
                break
            adj_vector = gwl.convert_into_adj_vector(query_data=query)
            if query in batch_dict.keys():
                print(query)
                break
            else:
                batch_dict[cnt] = [query, adj_vector, query_Train_labels[idx1 * configs["batch_size"] + idx2]]
                cnt += 1

        f = open(dirname + "train_adj_vectors_" + str(idx1) + ".p", "wb")
        pickle.dump(batch_dict, f)
        f.close()

    num_test_batch = int(len(query_Test_edges) / configs["batch_size"])

    for idx1 in range(num_test_batch):
        batch_dict = dict()
        cnt = 0
        for idx2 in range(configs["batch_size"]):
            if idx2 % 100 == 0:
                print("Encoding Test data - ", idx1, idx2)
            try:
                query = tuple(query_Test_edges[idx1 * configs["batch_size"] + idx2])
            except:
                break
            adj_vector = gwl.convert_into_adj_vector(query_data=query)
            if query in batch_dict.keys():
                print(query)
                break
            else:
                batch_dict[cnt] = [cnt, adj_vector, query_Test_labels[idx1 * configs["batch_size"] + idx2]]
                cnt += 1

        f = open(dirname + "test_adj_vectors_" + str(idx1) + ".p", "wb")
        pickle.dump(batch_dict, f)
        f.close()

    print ("Encoding - Elapsed time : ", time.time() - start)