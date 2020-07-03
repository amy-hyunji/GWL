# Generalized-WL

## Requirement 
- numpy 
- scipy
- pandas
- sklearn
- tensorflow 
- networkx

## How to Run the GWL
To run the GWL main.py, it is necessary to check the directory structure.  
Each directory's role and description are as follow:

- project_data : This directory stores project data which are given.
- input : This directory stores output of stage2 and preprocessed data for GWL.
  > final_unweighted_neg_sampling.csv : output of stage2, it should be stored in this directory. 
  > node2vec_unweighted.model : output of stage2, it should be stored in this directory.  
  > Other necessary data and directory for GWL will be automatically created by code. See the main.py & preprocessing.py & encoding.py   
- output : This directory stores configs, trained model, training loss and accuracy graph image, test result.  
  > Other necessary data and directory for GWL will be automatically created by code. See the training.py & testing.py  
- src : This directory stores stage3 codes

After construct the above directory structure, please run the **main.py**

## Note
If you want to use pretrained data to save time, please visit the under link.
https://drive.google.com/drive/folders/15MVK7QQy8Rudwb1zRZ6yDRYRcy3tn4x1?usp=sharing
- input/encoded_vector.zip : encoded vector of paper author, query public and answer public by using GWL with default configs.  
                             Unzip it and please locate at "./input/encoded_vector/n2v/1/npy/"
- output/model_small.h5 : pre-trained model which trained by paper author raw and sampled negative communities. Unzip it and please locate at "../output/n2v/1/npy/"
- output/model_big.h5 : pre-trained model which trained by aper author raw, sampled negative communities, query public and answer public. Unzip it and please locate at "../output/n2v/1/npy/"
                        
After construct the above directory structure, please run the **testing.py** to do the test with query public data. 
