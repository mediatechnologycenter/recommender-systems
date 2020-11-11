#Input Format
###File metadata.csv
Contains two columns:
* "resource_id": unique id of the article
* "text": full body article text without html tags. 
* "publication_date": date of publication. 

###File user_item_matrix.pq
Contains three columns:
* "user_id": unique id of the user
* "resource_id": unique id matching the one in metadata.csv
* "timestamp":  timestamp of the click

#Environement
Python 3.6.8

Packages in requirements.txt

#Usage
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
