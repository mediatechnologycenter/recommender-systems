This repository contains a federated learning approach for neural collaborative filtering (idea1) on news articles. We did not
find any existing open source code for federated neural collaborative filtering. This repository can be used to test the
novel approach described in [[1]](#1) which combines content information with the user history. The repo allows you to compare this approach against common recommender system models as baselines.

## Algorithm description
We expect the items to be text. The approach embeds the items via BERT into a vector space. The users
are than embedded via these item embeddings by averaging over all item vectors of items the user liked (in the training data).
For each user item pair we concatenate the user and item vector and feed it into a neural network.
We also sample a item the user did not like and feed it into the network. We then calculate the pairwise
loss between the positive and the negative sample and backpropagate. 

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
* "time":  timestamp of the click

# Get Started

## Environement

Python 3.6.8

Packages are in requirements.txt

## Usage
You can run `run_all.sh` to generate some dummy data and run all the algorithms. The jupyter notebook `results/evaluate.ipynb` displays the results. 

To work with **your own data** create a folder and copy the data in the format described above and run `DATA_FOLDER=your_data_folder python preprocessing.py` to preprocess your data.
You can then run any of the algorithms with `DATA_FOLDER=your_data_folder python algorithm.py`

The **results** are printed and saved in `results/{evaluation_name}`

The **training history** of the loss and metrics  are in `results/idea1_models/{model_name}`

Here is the description of the individual algorithms:

* content_based.py: Embeds articles with tf-idf, creates user-vectors by averaging over the articles. Creates a ranked
  list of recommendations by cos-similarity. To use your own embedding use
  `content_predict_custom_embedding`
* idea1.py: Embeds the article with a pretrained bert embedding. Embeds the users as the average of the article vectors 
  the user read. Trains a neural network with pairwise loss
  and creates predictions. (you can use your own embedding as well). For the decription of all the parameters look at
  chapter `Parameters for Idea1`
* popularity_random: Predicts simply the most popular articles. Predicts random articles.
* mf_model: Uses ALS to embed the articles and users as latent vectors. Then predicts the top articles for each user
  based on the latent vectors.
* folder example_scripts:
    * idea1_timewise_sampling.py is an example for the idea1 with timewise sampling instead of  random sampling
    * base_FL.py is an example for federated learning. Both users have the same data.
    * idea2.py is an example, where the corresponding user and item vector are not concatenated from the beginning but rather have some individual layers before the vectors are concatenated.

# Data Flow:

The flow through the system is as follows:

0. (_raw data_: The preprocessing transforms the data into horizontal format and generates a train, validation, test
   split. It stores the new data in the folder given by `DATA_FOLDER` ('processed' by default).
1. _formatted data_: load_data from the preprocessing module will load the formatted train,test and validation data.
2. _preprocessed/embedded data_: Each individual algorithm processes the data and prepares it for training. For Idea1
   this step embeds the users and articles into a vector representation and stores these lookup tables in the "
   processed"-folder.
3. _training_: Trains and outputs the trained model.
4. _prediction_: Takes the model and a list of users as input and returns a sorted list of recommendations for each
   user.
5. _evaluation_: Takes the predicted list of recommendations and the ground truth as input, and calculates the evaluation
   metrics.

# Modularity

In the Data Flow we see that every algorithm needs to load the data and evaluate on the data. These two steps are done
in the preprocessing resp. the evaluation module.

* preprocessing.py: Before we run any algorithm we first need to generate the formatted data. This is done by running
  this module as a script. After this is done, each algorithm simply calls preprocessing.load_data to load the
  formatted data.
* evaluation.py: This module expects two pd.Series of users:
    * _prediction_: containing a sorted list of article-IDs where the first item is the first ranked article for each
      user.
    * _ground_truth_: containing a list of article-IDs representing the actually read articles for each user.
      \
      The script calculates Recall@k for k=5,10,50,100 and NDCG@k for k=10,100 from these two pd.Series. **Make sure
      that the index of the two pd.Series match!**. It is the responsability of the prediction algorithm to exclude
      already read articles in the prediction.

# Parameters for Idea1
Main parameters:

|Parameter Name | Description | Default |
|-------|------------------------|---------|
|**lr**|Learning Rate for optimizer|0.00001|
|**batch_size**|Number of samples in each batch|100|
|**epochs**|Number of epochs to train|50|
|**layers**|Defines the layers and nodes in the layers. e.g. [a,b,c] will result in a 3 layer network with a nodes in layer 1, b nodes in layer 2, c nodes in layer 3. There is a second possible structure: [[a,b],[c,d] which means that the user and item vector first go through separate layers a(user) and b(user) resp. a(item), b(item) before they are concatenated. Afterwards the concatenated vector is run through layer c and d.|[1024, 512, 8]|
|**dropout**|Dropout value after each layer|0.5|
|**reg**|L2 Regularization applied to each layer. 0 Means no regularization.|0|
|**early_stopping**|Stop training if we do not see a decrease of the validation loss in the last `early_stopping` training rounds. 0 means no early stopping|0|
|**stop_on_metric**|If True then the early stopping criterion switches to the metrics in the evaluation step. i.e. stop if neither NDCG@100 nor Recall@10 from the validation set did increase in the last `early_stopping` training rounds|False|
|**random_sampling**|Whether to use random sampling (True) or timewise sampling (False). timewise sampling expects vertical format loaded with load_data_vertical and prepocessed negative samples|True|
|**folder**|Only used for timewise sampling. Working folder to store negative samples |None|

Other parameters:

|Parameter Name | Description | Default |
|-------|------------------------|---------|
|alpha|Proportion of pairwise loss compared to pointwise loss. Loss is calculated as alpha*pairwise_loss+(1-alpha)*pointwise loss|1|
|dropout_first|Dropout for the input|same as dropout|
|normalize|Type of normalization to apply. 0=no normalization. 1=normalize concatenated user and item vector together. 2=normalize user and item vector separatly|0|
|interval|Evaluation interval. Calculate metrics every `interval` epochs |1|
|checkpoint_interval|Store the model every `checkpoint_interval` epochs|1|
|loss|Type of pairwise loss to use. Can be either TOP or BPR|"BPR"|
|optimizer|What optimizer to use. Can be one "ADAM" or "SGD" |"ADAM"|
|take_target_out|If set to True then the vector of the current positive sample is taken out of the user vector.|False|
|workers|Number of workers to use for feeding the data to the network.|1|
|train|Whether to train the network (True) or simply load it (False)|True|
|round|Only used in Federated Learning. Current federated learning training round|False|
|epsilon|Only used in Federated Learning. Epsilon value to calculate the noise while training. 0 means no noise|0|

## References

<a id="1">[1]</a>
Wanyu Chen, Fei Cai, Honghui Chen, Maarten de Rijke (2019). Joint Neural Collaborative Filtering for Recommender Systems
https://arxiv.org/abs/1907.03459