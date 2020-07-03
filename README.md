# GWL 
codes for community detection and link prediction using EEWL and node2vec

## Datasets
#### Paper Author.txt
This dataset contains the true collaborations done in the past.
Authors         58646
Publications    137958
#### Query Public.txt
This dataset contains both the information of fake collaboration and real collaborations.
Authors         58646
Publications    34479

## Stage2
Generate fake collaboration dataset using rule-based Negative Sampling.
Node Embeddings are trained by Node2Vec in parallel.

## Stage3
Extract subgraph embedding vector based on the output of stage2.
Using simple classifer to detect whether the community is likely to be formed or not.
