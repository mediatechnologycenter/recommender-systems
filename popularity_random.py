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
    predictions = predictions[[x[1] not in x[0] for x in zip(predictions['read_articles'], predictions['predictions'])]]
    print(f"Pop prediction done in {datetime.datetime.now() - now}")

    return predictions


def random_predict(articles, grouped_train, N=10):
    now = datetime.datetime.now()
    predictions = pd.DataFrame(grouped_train)
    predictions['predictions']=predictions['article_id'].apply(lambda x:random.sample(list(articles), N))
        # prediction = random.sample(list(articles), N)
        # predictions[user_ix] = [list(prediction)]
    # predictions = pd.DataFrame(predictions).T
    # predictions.columns = ['predictions']

    predictions['read_articles'] = grouped_train
    isnull = predictions['read_articles'].isnull()
    if isnull.sum() > 0:
        predictions.loc[isnull, 'read_articles'] = [[[]] * isnull.sum()]
    predictions['predictions'] = [[article for article in x[1] if article not in x[0]] for x in zip(predictions['read_articles'], predictions['predictions'])]
    print(f"Pop prediction done in {datetime.datetime.now() - now}")

    return predictions
from helper import restrict_articles_to_timeframe

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

    pop = evaluate(pred, user_item_test.loc[pred.index], experiment_name='popularity_inital_results',limit=limit)

    random_scores = random_predict(itemids, user_item_train.head(limit), N)
    rand_algorithm = evaluate(random_scores, user_item_test.loc[pred.index],experiment_name='random_inital_results', limit=limit)

    b=pd.DataFrame(user_item_test)
    b.columns=['predictions']
    best_res = evaluate(b.loc[pred.index], user_item_test.loc[pred.index],experiment_name='best_res', limit=limit)
