from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from node2vec.edges import HadamardEmbedder
import pandas as pd
import random
import sys
from tqdm import tqdm

unweighted = Word2Vec.load("./n_unweighted.model")
weighted = Word2Vec.load("./n_weighted.model")
NODENUM = 58646
NEEDNUM = {2: 91241, 3: 31701, 6: 1087, 4: 8651, 5: 2662, 16: 96, 9: 262, 20: 65, 8: 384, 15: 80, 12: 155, 13: 128, 7: 588, 11: 184, 14: 121, 10: 215, 19: 60, 17: 70, 22: 24, 25: 27, 18: 58, 24: 36, 21: 50, 23: 24}
weight = True 
intList = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

def is_weighted():
    return weight

def _save_dict(_dict, _name):
    node1 = []
    node2 = []
    _keys = _dict.keys()
    for key in _keys:
        elem = _dict[key]
        node1.append(key)
        node2.append(elem)
    df = pd.DataFrame({"node1": node1, "node2": node2})
    df.to_csv(_name, index=False)

"""
return 
1. top 10 least similar nodes for each node {key: node1, val: list of 10 least similar nodes}
2. pick bidirectional ones from 1 {key: node1, val: nodes(smaller than node1) bidirectional to node1}
"""
def preprocess():
    nodeDict = dict() 
    biDict = dict()

    print("*** START preprocess ***")
    for i in tqdm(range(NODENUM)):
        node2 = []

        if (is_weighted()):
            vector1 = weighted.wv.most_similar(str(i), topn = NODENUM)
        else:
            vector1 = unweighted.wv.most_similar(str(i), topn = NODENUM)

        for m in range(10):
            lowest_similarity, val = vector1[-1-m]
            node2.append(str(lowest_similarity))
            # find if it is bidirectional
            if (str(lowest_similarity) in nodeDict.keys()):
                _temp = nodeDict[str(lowest_similarity)]
                if (str(i) in _temp):
                    # is bidirectional - add in node2 {key: str(i), val: str(lowest_similarity)}
                    if (str(i) not in biDict.keys()):
                        biDict[str(i)] = [str(lowest_similarity)]
                    else:
                        biDict[str(i)].append(str(lowest_similarity))
        # done 10 iters
        nodeDict[str(i)] = node2.copy()
    print("*** DONE preprocess ***")
    print("SAVING")

    # save
    if (is_weighted()):
        _save_dict(nodeDict, "weighted_neg_sample.csv")
        _save_dict(biDict, "bi_weighted_neg_sample.csv")
    else:
        _save_dict(nodeDict, "unweighted_neg_sample.csv")
        _save_dict(biDict, "bi_unweighted_neg_sample.csv")

    return nodeDict, biDict

"""
return list of bidirectional nodes
"""
def _get_bidirectional_list():
    if (is_weighted()):
        _bi = pd.read_csv("./bi_weighted_neg_sample.csv")
    else:
        _bi = pd.read_csv("./bi_unweighted_neg_sample.csv")
    bi_node1 = _bi["node1"]
    bi_node2 = _bi["node2"]
    retList = []
    for i in range(len(bi_node1)):
        _bi_node1 = str(bi_node1[i])
        _bi_node2 = str_to_list(bi_node2[i])
        for j in range(len(_bi_node2)):
            retList.append([_bi_node1, _bi_node2[j]])
    random.shuffle(retList)
    return retList

def str_to_list(_str):
    _str = _str.split("[")[1].split("]")[0]
    _str = _str.split("',")
    for i in range(len(_str)):
        elem = _str[i]
        for char in list(elem):
            if (char not in intList):
                _str[i] = _str[i].replace(char, '')
    return _str

def _get_neg_sample_list():
    if (is_weighted()):
        _sample = pd.read_csv("./weighted_neg_sample.csv")
    else:
        _sample = pd.read_csv("./unweighted_neg_sample.csv")

    sample_node1 = _sample["node1"]
    sample_node2 = _sample["node2"]
    retDict = dict() 
    for i in range(len(sample_node1)):
        retDict[str(sample_node1[i])] = str_to_list(sample_node2[i]).copy()
    return retDict

"""
input: list of nodes
output: node that is not likely to get connected with this list
"""
def _random_pick(_list, nodeDict):
    _idx = random.randint(1, len(_list))
    _elem = _list[_idx-1]
    candidateList = nodeDict[_elem]
    _idx = random.randint(1, len(candidateList))
    return candidateList[_idx-1]


def get_finalList(nodeDict):
    finalDict = dict()
    bi = _get_bidirectional_list()
    print("length of bidirectional elem: {}".format(len(bi))) # 120240
    if (len(nodeDict.keys()) == 0):
        nodeDict = _get_neg_sample_list()
    _biIndex = 0
    finalList = []

    for i in range(2, 26):
        _num = NEEDNUM[int(i)]
        print("**** Working on {} out of {} ****".format(i, '25'))
        # get 'j' number of new list with 'i' elements inside
        j = 0
        while (int(j) != int(_num)):
            print(" --- {} / {} --- ".format(str(j), _num))
            if(i == 2):
                _randStart = str(random.randint(1, NODENUM)-1)
                _randList = nodeDict[_randStart]
                _randEnd = random.randint(1, len(_randList))
                _bi = [_randStart, _randList[_randEnd-1]]
            else:
                _bi = bi[_biIndex]

            while (len(_bi) < int(i)):
                _randElem = _random_pick(_bi, nodeDict)
                _bi.append(_randElem)
            if (_bi in finalList):
                # in case of DUP
                continue
            else:
                if (i!=2):
                    _biIndex += 1
                finalList.append(_bi)
                j += 1
        print("**** SAVING ****")
        df = pd.DataFrame({"community": finalList})
        if (is_weighted()):
            df.to_csv("final_weighted_neg_sampling.csv", index=False)
        else:
            df.to_csv("final_unweighted_neg_sampling.csv", index=False)
    return 

def check():
    _file = pd.read_csv("./final_unweighted_neg_sampling.csv")
    _com = _file["community"]
    numDict = dict()
    for elem in _com:
        _elem = str_to_list(elem)
        _len = len(_elem)
        if (_len in numDict.keys()):
            numDict[_len] += 1
        else:
            numDict[_len] = 1

    _key = numDict.keys()
    if (_key != NEEDNUM.keys()):
        print("Oops.. keys are different.")
        print("keys for numDict: {}".format(_key))
        print("keys for NEEDNUM: {}".format(NEEDNUM.keys()))
    for elem in _key:
        if (numDict[elem] != NEEDNUM[elem]):
            print("[Elem: {}] neednum: {} vs. have: {}".format(elem, NEEDNUM[elem], numDict[elem]))

if __name__ == "__main__":
    nodeDict = dict()
    nodeDict, biDict = preprocess()
    get_finalList(nodeDict)
    check()
