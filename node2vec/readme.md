"""readme of node2vec"""

# Import
from node2vec.edges import HadamardEmbedder
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

# Load Model
>> unweighted = Word2Vec.load("com_embeddings.mdoel")
>> weighted = Word2Vec.load("weighted_embeddings.model")

# Get Embedding Vector (of test) 
>> test = "1" 
>> unweighted.wv.get_vector(test)   # vector of unweighted version
>> weighted.wv.get_vector(test)     # vector of weighted version

# Get Similarity of param1 and param2
>> param1 = "6350"
>> param2 = "2"
>> unweighted.wv.similarity(param1, param2)
>> weighted.wv.similarity(param1, param2)

# Get node with lowest similarity with 'test'
>> test = "1"
>> vector1 = unweighted.wv.most_similar(test, topn=58647)
>> (lowest_similarity_node, similarity_value) = vector1[-1]
