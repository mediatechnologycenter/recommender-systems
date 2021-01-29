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
import os
import implicit
import pandas as pd
from scipy import sparse


def mf_model(grouped_train, model_type='als', new=True, factors=50, model_path='mf_model.pkl'):
    """

    """
    now = datetime.datetime.now()
    grouped_train_exploded = grouped_train.explode().reset_index()
    grouped_train_exploded.columns = ['user', 'artist']
    grouped_train_exploded['user'] = grouped_train_exploded['user'].astype("category")
    grouped_train_exploded['artist'] = grouped_train_exploded['artist'].astype("category")
    grouped_train_exploded['rating'] = 1

    item_user_data = sparse.coo_matrix((grouped_train_exploded['rating'].astype(float),
                                        (grouped_train_exploded['artist'].cat.codes,
                                         grouped_train_exploded['user'].cat.codes)))
    # # initialize a model
    if new:
        if model_type == 'als':
            model = implicit.als.AlternatingLeastSquares(factors=factors,)
        elif model_type == 'bpr':
            model = implicit.bpr.BayesianPersonalizedRanking(factors=factors)
        model.fit(item_user_data)

        pickle.dump(model, open(model_path, "wb+"))
    else:
        model = pickle.load(open(model_path, "rb"))

    user_items = item_user_data.T.tocsr()
    print(f"MF training done in {datetime.datetime.now() - now}")
    return model, user_items, grouped_train_exploded


def mf_predict(model, user_items, grouped_train, N=10,article=None,users=None):
    now = datetime.datetime.now()
    predictions = {}

    for user_id, user_ix in dict(enumerate(grouped_train['user'].cat.categories)).items():
        if users is not None:
            if user_ix not in users.index:
                continue
        prediction = model.recommend(user_id, user_items, N=N, filter_already_liked_items=True)
        prediction = grouped_train['artist'].cat.categories[[x[0] for x in prediction]]
        if article is not None:
            prediction=[x for x in list(prediction) if x in article['resource_id']]
        predictions[user_ix] = [prediction]

    predictions = pd.DataFrame(predictions).T
    predictions.columns = ['predictions']
    print(f"MF prediction done in {datetime.datetime.now() - now}")

    return predictions


from preprocessing import load_data, get_metadata
from evaluation import evaluate
from helper import restrict_articles_to_timeframe

if __name__ == "__main__":
    N = 150  # number of predictions
    limit = 1000  # number of samples to look at
    factors = 20 # latent dim for ALS
    model_type = 'als'
    folder = os.getenv('PARTNER_A_OUTPUT_FOLDER','processed')

    user_item_train, user_item_test, user_item_validation = load_data(folder=folder, cut=40)
    metadata = get_metadata(folder=folder, usecols=['resource_id', 'text'])  # slow
    user_item_train = user_item_train.head(limit)
    user_item_test = user_item_test.head(limit)

    print(f"Data loaded")
    if not os.path.exists('mf_models'):
        os.makedirs('mf_models')
    model, user_items, grouped_train_exploded = mf_model(user_item_train.head(limit),
                                                                     model_type=model_type,
                                                                     new=True, factors=factors,
                                                                     model_path='mf_models/modelname.pkl')

    mf_pred = mf_predict(model, user_items, grouped_train_exploded.head(limit), N=N)
    mf_pred = mf_pred[mf_pred.index.isin(user_item_test.index)]

    mf = evaluate(mf_pred, user_item_test.loc[mf_pred.index], limit=limit, experiment_name='matrix_factorization.results')
