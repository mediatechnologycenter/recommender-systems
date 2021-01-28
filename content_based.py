# Copyright 2019-2020, ETH Zurich, Media Technology Center
#
# This file is part of Federated Learning Project at MTC.
#
# Federated Learning is a free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Federated Learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser Public License for more details.
#
# You should have received a copy of the GNU Lesser Public License
# along with Federated Learning.  If not, see <https://www.gnu.org/licenses/>.

import datetime
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
"""
These algorithms created predictions only based on the content. We have content_predict_tfidf which creates a tfidf 
embedding and content_predict_custom_embedding where you can input a custom embedding to calculate the cosin similarity 
with.
"""

def content_predict_tfidf(grouped_train, meta_data, N, limit=600):
    """
    Embeds each article with tf idf. Creats a Uservector for each user by averaging all embedded article vectors from
    articles that he read in past. Outputs the N articles that are closest(cosin similarity) to the uservector. We only
    consider articles that he didn't read in past.
    :param Series grouped_train: Index is the UserID, Values is a list of ArticleID of articles the user read.
    :param DataFrame meta_data: Contains article text and id. Requires two columns: "text" and "resource_id"
    :param N: Number of articles to return
    :param limit: number of users to look at.
    :return: Returns a DataFrame with UserID as index and a column "predictions" containing a list of N articles
             which the user did not yet read.
    """
    now = datetime.datetime.now()


    vectorizer = TfidfVectorizer(use_idf=True, max_features=10000)
    lookup = vectorizer.fit_transform(list(meta_data['text']))
    lookup = pd.DataFrame(lookup.T.todense(), index=vectorizer.get_feature_names(),
                          columns=meta_data['resource_id']).T

    if 'test' in meta_data.columns:
        lookup_test=lookup[lookup.index.isin(meta_data[meta_data['test']].resource_id)]
    else:
        lookup_test=lookup

    grouped_train = grouped_train.head(limit)

    def get_user_similarity(user_history, N=10):
        uservector = lookup.loc[user_history, :].mean() * vectorizer.idf_
        prediction = lookup_test.dot(uservector)
        top_10_other = prediction.drop(index=user_history, errors='ignore').sort_values(ascending=False).iloc[:N].reset_index()

        return top_10_other['resource_id'].tolist()

    predictions = grouped_train.apply(lambda x: get_user_similarity(x, N))
    print(f"Prediction done in {datetime.datetime.now() - now}")
    predictions = pd.DataFrame(predictions)
    predictions.columns = ['predictions']
    return predictions

def content_predict_custom_embedding(train,article_embedding,user_embedding, N):
    """
    Outputs the N articles that are closest(cosin similarity) to the uservector. We only
    consider articles that he didn't read in past.
    :param Series train: Index is the UserID, Values is a list of ArticleID of articles the user read.
    :param DataFrame article_embedding: Index is the ArticleID, columns are embedding-dimensions.
    :param DataFrame user_embedding: Index is the UserID, columns are embedding-dimensions.
    :param N: Number of articles to return
    :return: Returns a DataFrame with UserID as index and a column "predictions" containing a list of N articles
             which the user did not yet read.
    """
    now = datetime.datetime.now()

    def get_user_similarity(user, N=10):
        uservector = user_embedding.loc[user, :]
        uservector.index=article_embedding.columns
        prediction = article_embedding.dot(uservector)
        top_10_other = prediction.sort_values(ascending=False).iloc[:N]

        return top_10_other.index.tolist()

    prediction = train.reset_index()['user_ix'].apply(lambda x: get_user_similarity(x, N))

    prediction = [[article for article in x[1] if article not in x[0]] for x in
                                  zip(train, prediction)]

    print(f"Prediction done in {datetime.datetime.now() - now}")
    predictions = pd.DataFrame(pd.Series(prediction))

    predictions.columns = ['predictions']
    predictions.index=train.index
    return predictions

from preprocessing import load_data, get_metadata
from evaluation import evaluate
from helper import restrict_articles_to_timeframe

if __name__ == "__main__":
    N = 150  # number of predictions
    limit = 10000  # number of samples to look at
    folder = os.getenv('PARTNER_A_OUTPUT_FOLDER','processed')

    user_item_train, user_item_test, user_item_validation = load_data(folder=folder)
    meta_data = get_metadata(folder=folder,usecols=['resource_id','text'])

    content_based_prediction = content_predict_tfidf(user_item_train, meta_data, N, limit=limit).sort_index()
    content_based_prediction = content_based_prediction[content_based_prediction.index.isin(user_item_test.index)]
    content = evaluate(content_based_prediction, user_item_test.loc[content_based_prediction.index], experiment_name='tf_idf.results',limit=limit)


    # Tipp: with content_predict_custom_embedding you can input a custom embedding for users/articles.