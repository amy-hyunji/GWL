'''
@ author: Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

from preprocessing import preprocessing
from encoding import encoding
from training import training
from testing import testing
from pickle2npy import pickle2npy
import os

configs = {
        "is_raw" : True,
        "basedir" : "../input/",
        "graph": "link combination of paper author",
        "pos communities": "paper author raw",
        "neg communities": "negative sampled community",
        "Test": "query public raw",
        "h_size": 1,
        "exist_node2vec": True,
        "batch_size": 1000,
        "target": "raw",  # "raw" or "edges"
        "total_epochs": 50,
        "type_n2v": "n2v",  # "n2v" or "nn2v"
        "n_hidden": [128, 32, 8, 2],
        "data_type" : "npy" # "npy" or "pickle"
    }
input_dir = configs["basedir"]
for i in ["encoded_vector", configs["type_n2v"], str(configs["h_size"]), configs["data_type"]]:
        input_dir += i+"/"
        if not os.path.isdir(input_dir):
                os.mkdir(input_dir)

output_dir = "../output/"
if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
for i in [configs["type_n2v"], str(configs["h_size"]), configs["data_type"]]:
        output_dir += i + "/"
        if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

#preprocessing(configs)
#encoding(configs)
#if configs["data_type"] == "npy":
#        pickle2npy(configs)
training(configs)
testing(configs)

