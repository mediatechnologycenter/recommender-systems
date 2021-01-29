#   Copyright 2021 ETH Zurich, Media Technology Center
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import datetime
import pickle

import implicit
import pandas as pd
from scipy import sparse
import os
import sys

sys.path.append(os.getcwd())
from preprocessing import load_data_cv, transform_horizontal_to_vertical, get_metadata
from evaluation import evaluate
import random

def pop_predict(model, grouped_train, N=10):
    now = datetime.datetime.now()
    prediction=model[:N]['article_id'].values.tolist()
    predictions = pd.DataFrame(grouped_train)
    predictions['predictions'] = pd.Series([prediction] * len(predictions) ,index=predictions.index)

    predictions['read_articles'] = grouped_train
    isnull = predictions['read_articles'].isnull()
    if isnull.sum() > 0:
        predictions.loc[isnull, 'read_articles'] = [[[]] * isnull.sum()]
    predictions['predictions'] = [[article for article in x[1] if article not in x[0]] for x in zip(predictions['read_articles'], predictions['predictions'])]
    print(f"Pop prediction done in {datetime.datetime.now() - now}")

    return predictions


def random_predict(articles, grouped_train, N=10):
    now = datetime.datetime.now()
    predictions = pd.DataFrame(grouped_train)
    predictions['predictions']=predictions['article_id'].apply(lambda x:random.sample(list(articles), N))

    predictions['read_articles'] = grouped_train
    isnull = predictions['read_articles'].isnull()
    if isnull.sum() > 0:
        predictions.loc[isnull, 'read_articles'] = [[[]] * isnull.sum()]
    predictions['predictions'] = [[article for article in x[1] if article not in x[0]] for x in zip(predictions['read_articles'], predictions['predictions'])]
    print(f"Pop prediction done in {datetime.datetime.now() - now}")

    return predictions

if __name__ == "__main__":
    N = 50 #number of predictions
    limit = 10000 # number of samples to look at


    folder = os.getenv('PARTNER_A_OUTPUT_FOLDER','processed')
    user_item_train, user_item_test, user_item_validation = load_data_cv(folder=folder)
    user_item_train2 = transform_horizontal_to_vertical(user_item_train)
    itemids = user_item_train2['article_id'].unique()
    popularity = user_item_train2.groupby('article_id').count().reset_index().sort_values(
        "user_ix", ascending=False)

    pred = pop_predict(popularity, user_item_train.head(limit), N)

    pop = evaluate(pred, user_item_test.loc[pred.index], experiment_name='popularity.results',limit=limit)

    random_scores = random_predict(itemids, user_item_train.head(limit), N)
    rand_algorithm = evaluate(random_scores, user_item_test.loc[pred.index],experiment_name='random.results', limit=limit)

    b=pd.DataFrame(user_item_test)
    b.columns=['predictions']
    best_res = evaluate(b.loc[pred.index], user_item_test.loc[pred.index],experiment_name='best.results', limit=limit)
