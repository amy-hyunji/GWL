'''
@ author: Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

import pickle
from itertools import combinations
import os
import csv

def preprocessing(configs):
    '''
    preprocess the given data & save as pickle
    '''
    is_raw = configs["is_raw"]
    basedir = configs["basedir"]

    ''' paper author edges set '''
    f = open("../project_data/paper_author.txt", "r")
    data = f.readlines()
    f.close()
    new_list = list(map(lambda x: x.strip().split(' '), data[1:]))

    edges = []
    for idx in range(len(new_list)):
        onedata = list(map(lambda x: int(x), new_list[idx]))
        oneedges = list(combinations(onedata, 2))
        edges += oneedges

    f = open(basedir+"pa_edges_set.p", "wb")
    pickle.dump(set(edges), f)
    f.close()
    print ("Preprocessing - Saved pa_edges_set.p")

    ''' query public and answer public '''
    f = open("../project_data/query_public.txt", "r")
    _edge_data = f.readlines()
    f.close()

    f = open("../project_data/answer_public.txt", "r")
    _label_data = f.readlines()
    f.close()

    edge_data = _edge_data[1:]
    data_split = list(map(lambda x: x.strip().split(' '), edge_data))
    label_data = list(map(lambda x: x.strip(), _label_data))

    edges_raw = []
    labels_raw = []

    for idx in range(len(data_split)):
        onedata = list(map(lambda x: int(x), data_split[idx]))
        onedata_sorted_tuple = tuple(sorted(onedata))

        if onedata_sorted_tuple not in edges_raw:
            edges_raw.append(onedata_sorted_tuple)
            if label_data[idx] == 'True':
                labels_raw.append(1)
            elif label_data[idx] == 'False':
                labels_raw.append(0)
            else:
                pass

    f = open(basedir+"qpu_edges_raw_duplicate.p", "wb")
    pickle.dump(edges_raw, f)
    f.close()
    print("Preprocessing - Saved qpu_edges_raw_duplicate.p")

    f = open(basedir+"apu_labels_raw_duplicate.p", "wb")
    pickle.dump(labels_raw, f)
    f.close()
    print("Preprocessing - Saved apu_labels_raw_duplicate.p")

    ''' query public and answer public '''

    f = open("../project_data/query_private.txt", "r")
    _edge_data = f.readlines()
    f.close()

    edge_data = _edge_data[1:]
    data_split = list(map(lambda x: x.strip().split(' '), edge_data))

    edges_raw = []
    for idx in range(len(data_split)):
        onedata = list(map(lambda x: int(x), data_split[idx]))
        onedata_sorted_tuple = tuple(sorted(onedata))
        edges_raw.append(onedata_sorted_tuple)

    f = open(basedir + "qpr_edges_raw_duplicate.p", "wb")
    pickle.dump(edges_raw, f)
    f.close()
    print("Preprocessing - Saved qpr_edges_raw_duplicate.p")

    ''' Positive Communities '''
    if is_raw:
        pa_raw = []
        for row in new_list:
            pa_raw.append(list(map(lambda x: int(x), row)))
        pa_raw_data = {"type": "list of list", "source": "preprocessing.py", "data": pa_raw}

        outdir = basedir+"raw/"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        f = open(outdir+"pos_communities.p", "wb")
        pickle.dump(pa_raw_data, f)
        f.close()

    else:
        pa_edges = []
        for row in new_list:
            row = list(map(lambda x: int(x), row))
            edges = list(combinations(row, 2))
            edges = list(map(lambda x: list(x), edges))
            pa_edges += edges
        pa_edges_data = {"type" : "list of list", "source" : "preprocessing.py", "data" : pa_edges}

        outdir = basedir + "edges/"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        f = open(outdir+"pos_communities.p", "wb")
        pickle.dump(pa_edges_data, f)
        f.close()
        print("Preprocessing - Saved pos_communities.p")

    ''' Negative Communities '''
    f = open(basedir+'final_unweighted_neg_sampling.csv', newline='')
    reader = csv.reader(f)

    data = []
    is_first = True
    for row in reader:
        if is_first:
            is_first = False
        else:
            row_split = row[0].split(', ')
            row_split[0] = row_split[0][1:]
            row_split[-1] = row_split[-1][:-1]
            neg_community = list(map(lambda x: int(x[1:-1]), row_split))
            data.append(neg_community)
    f.close()

    if is_raw:
        data = {"type": "list of list", "source": "preprocessing.py","data": data}

        f = open(outdir+"neg_communities.p", "wb")
        pickle.dump(data, f)
        f.close()
    else:
        neg_edges = []
        for row in data:
            edges = list(combinations(row, 2))
            edges = list(map(lambda x: list(x), edges))
            neg_edges += edges

        neg_edges_data = {"type" : "list of list", "source" : "preprocessing.py", "data" : neg_edges}

        f = open(outdir+"neg_communities.p", "wb")
        pickle.dump(neg_edges_data, f)
        f.close()
        print("Preprocessing - Saved neg_communities.p")

if __name__ == "__main__":
    is_raw = False
    basedir = "../input/"
    preprocessing(is_raw=is_raw, basedir=basedir)