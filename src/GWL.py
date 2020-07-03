'''
@ author: Dong-Hyeok Shin; tlsehdgur0@kaist.ac.kr
'''

import networkx as nx
from itertools import combinations
from scipy.stats.mstats import gmean
import numpy as np

class GWL:
    def __init__(self, graph, configs):
        self.graph = graph
        self.h_size = configs["h_size"]
        self.exist_node2vec = configs["exist_node2vec"]
        if self.exist_node2vec:
            from gensim.models import Word2Vec
            self.node2vec_model = Word2Vec.load(configs["basedir"]+"node2vec_unweighted.model")
        else:
            self.node2vec_model = None

    def find_subgraph_nodes(self, query):
        subgraph_nodes = set()
        for idx_node in query:
            length = nx.single_source_shortest_path_length(self.graph, idx_node,self.h_size)
            neighbors = list(length.keys())
            subgraph_nodes = subgraph_nodes | set(neighbors)
        return list(subgraph_nodes)

    def extract_subgraph(self, query):
        subgraph_nodes = self.find_subgraph_nodes(query)
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        return subgraph

    def compute_geometric_mean(self, subgraph, query):
        tmp_subgraph = subgraph.copy()

        ''' Remove combination edges in query '''
        query_edges = list(combinations(query, 2))
        for remove_edge in query_edges:
            try:
                tmp_subgraph.remove_edge(remove_edge[0], remove_edge[1])
            except:
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
                    if self.exist_node2vec:
                        query_node_vector = self.node2vec_model.wv.get_vector(str(query_node))
                        idx_node_vector = self.node2vec_model.wv.get_vector(str(idx_node))
                        shortest_path_lengths.append(np.linalg.norm(query_node_vector - idx_node_vector))
                    else:
                        shortest_path_lengths.append(nx.shortest_path_length(tmp_subgraph, source=query_node, target=idx_node))
                else:
                    shortest_path_lengths.append(n_nodes ** 2)
            tmp_subgraph.nodes[idx_node]['avg_dist'] = gmean(shortest_path_lengths)

        for query_node in query:
            tmp_subgraph.nodes[query_node]['avg_dist'] = 0.0

        return tmp_subgraph

    def sample(self, subgraph, nodelist, query):
        adj_matrix = np.asarray(nx.adj_matrix(subgraph, nodelist=nodelist).todense())
        vector = np.array([])
        for idx in range(len(adj_matrix)):
            if idx < len(query):
                vector = np.append(vector, adj_matrix[idx, len(query):])
            elif idx == len(adj_matrix) - 1:
                pass
            else:
                vector = np.append(vector, adj_matrix[idx, idx + 1:])
        return vector

    def convert_into_adj_vector(self, query_data):
        query = query_data
        subgraph = self.extract_subgraph(query=query)
        subgraph = self.compute_geometric_mean(subgraph=subgraph, query=query)

        avg_dist = nx.get_node_attributes(subgraph, 'avg_dist')

        _nodelist = sorted(avg_dist.items(), key=(lambda x: x[1]))
        nodelist = list(map(lambda x: x[0], _nodelist))
        truncate_size = len( query) * 10 * self.h_size
        if len(nodelist) > truncate_size:
            nodelist = nodelist[:truncate_size]
        adj_vector = self.sample(subgraph, nodelist, query)
        return adj_vector




