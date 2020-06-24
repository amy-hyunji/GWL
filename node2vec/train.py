import numpy as np
import networkx as nx
from node2vec import Node2Vec
import pickle

weighted = True

# create a graph
if not (weighted):
    with open('./network_graph.p', 'rb') as file:
        G = pickle.load(file)

else:
    f = open("./paper_author.txt")
    lines = f.readlines()
    G = nx.Graph()
    countDict = dict()

    for line in lines:
        _elem = line.split(" ")
        _elem[-1] = _elem[-1].split("\n")[0]
        _len = len(_elem)
        for i in range(_len-1):
            node1 = int(_elem[i])-1
            node2 = int(_elem[i+1])-1
            if (node1 > node2):
                appendTup = (node2, node1)
            else:
                appendTup = (node1, node2)

            if (appendTup in countDict):
                countDict[appendTup] += 1
            else:
                countDict[appendTup] = 1
    
    _keys = countDict.keys()
    for key in _keys:
        _key = list(key)
        G.add_edge(_key[0], _key[1], weight=countDict[key])
        
    tempList = []
    for i in range(58646):
        tempList.append(i)
    for node_idx in range(len(tempList)):
        if node_idx not in list(G.nodes):
            G.add_node(node_idx)

node2vec = Node2Vec(G, workers=8)

model = node2vec.fit(window=10, min_count=1, batch_words=4)

# save embeddings for later use
model.wv.save_word2vec_format("./n_weighted.emb")

# save model for later use
model.save("./n_weighted.model")

