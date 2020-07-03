# GWL 
codes for GWL, a method for community prediction scalable for all size of communities

## Datasets
#### Paper Author.txt
This dataset contains the true collaborations done in the past.<br/>
Authors         58646<br/>
Publications    137958
#### Query Public.txt
This dataset contains both the information of fake collaboration and real collaborations.<br/>
Authors         58646<br/>
Publications    34479

## Stage2
Generate fake collaboration dataset using rule-based Negative Sampling.<br/>
Node Embeddings are trained by Node2Vec in parallel.

## Stage3
Extract subgraph embedding vector based on the output of stage2.<br/>
Using simple classifer to detect whether the community is likely to be formed or not.
