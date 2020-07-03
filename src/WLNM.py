'''
@ author: Kien Mai Ngoc; https://github.com/KienMN/Weisfeiler-Lehman-Neural-Machine
@ modified : Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

import networkx as nx
import numpy as np
import math
import pandas as pd
import pickle
from functools import partial
import time
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import os

def treat_Warning1(input):
    ret = []
    for row in input:
        new_row = []
        for i in row:
            new_row.append(i - 1)
        ret.append(tuple(new_row))
    return ret

def enclosing_subgraph(fringe, network, subgraph, distance):
    neighbor_links = []
    for link in fringe:
        u = link[0]
        v = link[1]
        neighbor_links = neighbor_links + list(network.edges(u))
        neighbor_links = neighbor_links + list(network.edges(v))
    tmp_subgraph = subgraph.copy()
    tmp_subgraph.add_edges_from(neighbor_links)
    # Remove duplicate and existed edge
    neighbor_links = [li for li in tmp_subgraph.edges() if li not in subgraph.edges()]
    tmp_subgraph = subgraph.copy()
    tmp_subgraph.add_edges_from(neighbor_links, distance=distance, inverse_distance=1/distance)
    return neighbor_links, tmp_subgraph


def compute_geometric_mean_distance(subgraph, link):
    u = link[0]
    v = link[1]
    subgraph.remove_edge(u, v)

    n_nodes = subgraph.number_of_nodes()
    u_reachable = nx.descendants(subgraph, source=u)
    v_reachable = nx.descendants(subgraph, source=v)
    for node in subgraph.nodes:
        distance_to_u = 0
        distance_to_v = 0
        if node != u:
            distance_to_u = nx.shortest_path_length(subgraph, source=node,
                                                    target=u) if node in u_reachable else 2 ** n_nodes
        if node != v:
            distance_to_v = nx.shortest_path_length(subgraph, source=node,
                                                    target=v) if node in v_reachable else 2 ** n_nodes
        subgraph.nodes[node]['avg_dist'] = math.sqrt(distance_to_u * distance_to_v)

    subgraph.add_edge(u, v, distance=0)

    return subgraph

def prime(x):
    if x < 2:
        return False
    if x == 2 or x == 3:
        return True
    for i in range(2, x):
        if x % i == 0:
            return False
    return True


def palette_wl(subgraph, link, prime_numbers):
    tmp_subgraph = subgraph.copy()
    if tmp_subgraph.has_edge(link[0], link[1]):
        tmp_subgraph.remove_edge(link[0], link[1])
    avg_dist = nx.get_node_attributes(tmp_subgraph, 'avg_dist')

    df = pd.DataFrame.from_dict(avg_dist, orient='index', columns=['hash_value'])
    df = df.sort_index()
    df['order'] = df['hash_value'].rank(axis=0, method='min').astype(np.int)
    df['previous_order'] = np.zeros(df.shape[0], dtype=np.int)
    adj_matrix = nx.adj_matrix(tmp_subgraph, nodelist=sorted(tmp_subgraph.nodes)).todense()
    while any(df.order != df.previous_order):
        df['log_prime'] = np.log(prime_numbers[df['order'].values])
        total_log_primes = np.ceil(np.sum(df.log_prime.values))
        df['hash_value'] = adj_matrix * df.log_prime.values.reshape(-1, 1) / total_log_primes + df.order.values.reshape(
            -1, 1)
        df.previous_order = df.order
        df.order = df.hash_value.rank(axis=0, method='min').astype(np.int)
    nodelist = df.order.sort_values().index.values
    return nodelist

def extract_enclosing_subgraph(link, network, size=10):
    fringe = [link]
    subgraph = nx.Graph()
    distance = 0
    subgraph.add_edge(link[0], link[1], distance=distance)
    while subgraph.number_of_nodes() < size and len(fringe) > 0:
        distance += 1
        fringe, subgraph = enclosing_subgraph(fringe, network, subgraph, distance)

    tmp_subgraph = network.subgraph(subgraph.nodes)
    additional_edges = [li for li in tmp_subgraph.edges if li not in subgraph.edges]
    subgraph.add_edges_from(additional_edges, distance=distance + 1, inverse_distance=1 / (distance + 1))
    return subgraph

def sample(subgraph, nodelist, weight='weight', size=10):
    adj_matrix = nx.adj_matrix(subgraph, weight=weight, nodelist=nodelist).todense()
    vector = np.asarray(adj_matrix)[np.triu_indices(len(adj_matrix), k=1)]
    d = size * (size - 1) // 2
    if len(vector) < d:
        vector = np.append(vector, np.zeros(d - len(vector)))
    return vector[1:]### Subgraph encoding test

def encode_link(link, network, weight='weight', size=10):
    e_subgraph = extract_enclosing_subgraph(link, network, size=size)
    e_subgraph = compute_geometric_mean_distance(e_subgraph, link)

    prime_numbers = np.array([i for i in range(10000) if prime(i)], dtype=np.int)
    nodelist = palette_wl(e_subgraph, link, prime_numbers)
    if len(nodelist) > size:
        nodelist = nodelist[:size]
        e_subgraph = e_subgraph.subgraph(nodelist)
        nodelist = palette_wl(e_subgraph, link, prime_numbers)
    embeded_link = sample(e_subgraph, nodelist, weight=weight, size=size)
    return embeded_link


def show_result(Test_labels, predictions):
    ### Show AUC ###
    fpr, tpr, thresholds = metrics.roc_curve(Test_labels, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(auc)

    ### Show Precision, recall, F1_score ###
    precision = precision_score(Test_labels, predictions)  # tp/(tp+fp)
    recall = recall_score(Test_labels, predictions)  # tp/(tp+fn)
    F1_score = f1_score(Test_labels, predictions)

    print("precision : ", precision)
    print("recall : ", recall)
    print("F1_score : ", F1_score)

    ### Hard counting ###
    assert len(Test_labels) == len(predictions)

    cnt = 0
    for i in range(len(Test_labels)):
        if Test_labels[i] != predictions[i]:
            cnt += 1
    print('# of test links : ', len(Test_labels))
    print("# of missclassification : ", cnt)
    return auc, precision, recall, F1_score, cnt

if __name__ == "__main__":
    configs = {
        "is_raw": True,
        "basedir": "../input/",
        "graph": "link combination of paper author",
        "pos communities": "paper author raw",
        "neg communities": "negative sampled community",
        "Test": "query public raw",
        "h_size": 1,
        "exist_node2vec": False,
        "batch_size": 1000,
        "target": "raw",  # "raw" or "edges"
        "total_epochs": 50,
        "type_n2v": "nn2v",  # "n2v" or "nn2v"
        "n_hidden": [128, 32, 8, 2],
        "data_type": "npy"  # "npy" or "pickle"
    }

    outdirname = "../output/WLNM/"
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)

    ########## Bulid the Graph ##########

    f = open("../input/pa_edges_set.p", "rb")  # set of tuple
    pa_edges_set = pickle.load(f)
    f.close()

    n_nodes = max(list(map(lambda x: max(x), pa_edges_set)))
    n_links = len(pa_edges_set)
    print("# of total nodes : ", n_nodes)
    print("# of total links : ", n_links)

    graph_edges = treat_Warning1(pa_edges_set)

    network = nx.empty_graph()
    network.add_edges_from(graph_edges)

    for node_idx in range(n_nodes):
        if node_idx not in list(network.nodes):
            network.add_node(node_idx)

    ########## Load Train data ##########

    f = open("../input/"+configs["target"]+"pos_communities.p", "rb")
    _pos_communities = pickle.load(f)
    f.close()
    pos_communities = _pos_communities["data"]
    pos_communities = treat_Warning1(pos_communities)

    f = open("../input/"+configs["target"]+"neg_communities.p", "rb")
    _neg_communities = pickle.load(f)
    f.close()
    neg_communities = _neg_communities["data"]

    query_Train_edges = pos_communities + neg_communities
    query_Train_labels = [1] * len(pos_communities) + [0] * len(neg_communities)

    query_Train_edges, query_Train_labels = shuffle(query_Train_edges, query_Train_labels)

    print("# of Train edges : ", len(query_Train_edges))
    prime_numbers = np.array([i for i in range(10000) if prime(i)], dtype=np.int)

    f1 = partial(encode_link, network=network, weight='weight', size=10)
    ############################################
    start = time.time()
    num_train_batch = int(len(query_Train_edges) / configs["batch_size"])
    for idx1 in range(num_train_batch + 1):
        batch_data = "None"
        batch_label = []
        for idx2 in range(configs["batch_size"]):
            try:
                query = query_Train_edges[idx1 * configs["batch_size"]+idx2]
            except:
                break
            else:
                adj_vector = f1(query)
                if batch_data == "None":
                    batch_data = adj_vector
                else:
                    batch_data = np.vstack( (batch_data, adj_vector) )
                batch_label.append(query_Train_labels[idx1 * configs["batch_size"]+idx2])

        np.save(outdirname+"encoded_vector/train_adj_vector"+str(idx1), batch_data)
        np.save(outdirname + "encoded_vector/train_adj_label" + str(idx1), np.array(batch_label))

    ########## Load Test data ##########

    f = open("../input/qpu_edges_raw_duplicate.p", 'rb')
    _query_Test_edges = pickle.load(f)
    f.close()
    _query_Test_edges = treat_Warning1(_query_Test_edges)

    f = open("../input/apu_labels_raw_duplicate.p", 'rb')
    _query_Test_labels = pickle.load(f)
    f.close()

    if configs["target"] == "_links":
        query_Test_edges = []
        query_Test_labels = []
        for idx in range(len(_query_Test_edges)):
            if len(_query_Test_edges[idx]) == 2:
                query_Test_edges.append(_query_Test_edges[idx])
                query_Test_labels.append(_query_Test_labels[idx])
    else:
        query_Test_edges = _query_Test_edges
        query_Test_labels = _query_Test_labels

    print("# of Test edges : ", len(query_Test_edges))

    num_test_batch = int(len(query_Test_edges) / configs["batch_size"])
    for idx1 in range(num_test_batch + 1):
        batch_data = "None"
        for idx2 in range(configs["batch_size"]):
            try:
                query = query_Test_edges[idx1 * configs["batch_size"] + idx2]
            except:
                break
            else:
                adj_vector = f1(query)
                if batch_data == "None":
                    batch_data = adj_vector
                else:
                    batch_data = np.vstack((batch_data, adj_vector))

        np.save(outdirname + "encoded_vector/test_adj_vector" + str(idx1), batch_data)

