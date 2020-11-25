# Input Format
### File metadata.csv
Contains two columns:
* "resource_id": unique id of the article
* "text": full body article text without html tags. 
* "publication_date": date of publication. 

### File user_item_matrix.pq
Contains three columns:
* "user_id": unique id of the user
* "resource_id": unique id matching the one in metadata.csv
* "timestamp":  timestamp of the click

# Environement
Python 3.6.8

Packages in requirements.txt

# Usage
Run preprocessing.py to generate some random data or create data yourself. The functions in the module should describe 
the desired format and location of the files.
Run any of the algortihms:
* content_based.py: Embeds articles with tf-idf, creates user-vectors by averaging over the articles. Creates a ranked 
                    list of recommendations by cos-similarity. To use your own embedding use 
                    `content_predict_custom_embedding`
* idea1.py: Embeds the article and users with a pretrained bert embedding. Trains a neural network with pairwise loss 
            and creates predictions. (you can use your own embedding as well)
* popularity_random: Predicts simply the most popular articles. Predicts random articles. 

The results are printed and saved in `results/{evaluation_name}` 

# Data Flow:
The flow through the system is as follows:
0. (_raw data_: The preprocessing module will generate raw data from a given input folder resp. 
transform some data into the formatted data which is described in [Input Format](#Input Format).
It also generates an train, validation, test split and stores the data in the "processed"-folder.
1. _formatted data_: load_data from the preprocessing module will load the formatted train,test and validation data.
2. _preprocessed/embedded data_: Each individual algorithm processes the data and prepares it
for training. For Idea1 this step embeds the users and articles into a vector representation and 
stores these lookup tables in the "processed"-folder. 
3. _training_: Trains and outputs the trained model.
4. _prediction_: Takes the model and a list of users as input and returns a sorted list of recommendations for each user.
5. _evaluation_: Takes the predicted list of recommendation and the ground truth as input and
calculates the evaluation metrics.

# Modularity
In the Data Flow we see that every algorithm needs to load the data and evaluate on the data.
These two steps are done in the preprocessing resp. the evaluation module.
* preprocessing.py: Before we run any algorithm we first need to generate the formatted 
data. This is done by running this module as a script. After this is done, each algorithm 
simply calls preprocessing.load_data function to load the formatted data.
* evaluation.py: This module expects two pd.Series of users: 
    * _prediction_: containing a sorted list of article-IDs where the first item is
    the first ranked article for each user.
    * _ground_truth_: containing a list of article-IDs representing the actually read
     articles for each user. 
     \
The script calculates Recall@k for k=5,10,50,100 and NDCG@k for k=10,100 from these two 
pd.Series. **Make sure that the index of the two pd.Series match!**. It is the 
responsability of the prediction algorithm to exclude already read articles in 
the prediction.
