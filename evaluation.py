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
import numpy as np
import pandas as pd
"""
This module is used to evaluate the prediction of a model.
"""

def evaluate(pred, ground_truth, experiment_name='results.csv', limit=6000):
    """
    Calculates Recall@5,10,20,50 and NDCG@10,100 from the input. Careful! Due to performance issues, we do not check if
    the UserID of pred and ground_truth match. This means that pred and ground_truth need the same index IN THE SAME
    ORDER!
    :param pred: pd.DataFrame with index the UserID and one column: "predictions" which is a ranked list of ArticleIDs
                 First element is the article with highest rank.
    :param ground_truth: pd.Series with Index the UserID and value the list of ArticleIDs the user read.
    :param experiment_name: Output-path of the result. All results are APPENDED to the file results/experiment_name
    :param limit: Restrict the number of users to evaluate on to "limit"
    :return: Prints the evaluation metrics and stores result in results/experiment_name
    """


    r5_list = Recall_at_k_batch(pred, ground_truth, k=5, limit=limit)
    r10_list = Recall_at_k_batch(pred, ground_truth, k=10, limit=limit)
    r20_list = Recall_at_k_batch(pred, ground_truth, k=20, limit=limit)
    r50_list = Recall_at_k_batch(pred, ground_truth, k=50, limit=limit)
    n10_list = NDCG_binary_at_k_batch(pred, ground_truth, k=10, limit=limit)
    n100_list = NDCG_binary_at_k_batch(pred, ground_truth, k=100, limit=limit)

    print("Test NDCG@10=%.5f (%.5f)" % (np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))))
    print("Test Recall@5=%.5f (%.5f)" % (np.mean(r5_list), np.std(r5_list) / np.sqrt(len(r5_list))))
    print("Test Recall@10=%.5f (%.5f)" % (np.mean(r10_list), np.std(r10_list) / np.sqrt(len(r10_list))))
    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
    results = pd.DataFrame([["Test NDCG@10=%.5f (%.5f)", np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))],
                            ["Test Recall@5=%.5f (%.5f)", np.mean(r5_list), np.std(r5_list) / np.sqrt(len(r5_list))],
                            ["Test Recall@10=%.5f (%.5f)", np.mean(r10_list),
                             np.std(r10_list) / np.sqrt(len(r10_list))],
                            ["Test NDCG@100=%.5f (%.5f)", np.mean(n100_list),
                             np.std(n100_list) / np.sqrt(len(n100_list))],
                            ["Test Recall@20=%.5f (%.5f)", np.mean(r20_list),
                             np.std(r20_list) / np.sqrt(len(r20_list))],
                            ["Test Recall@50=%.5f (%.5f)", np.mean(r50_list),
                             np.std(r50_list) / np.sqrt(len(r50_list))]])

    if not os.path.exists('results'):
        os.makedirs('results')
    results.to_csv(f"results/{experiment_name}", mode='+a')

    return r20_list, r50_list, n100_list


def NDCG_binary_at_k_batch(predictions, ground_truth, k=100, limit=1000):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    predictions = predictions.head(limit)
    now = datetime.datetime.now()
    predictions['ground_truth'] = ground_truth.head(limit).tolist()
    DCG = [sum([1 / np.log2(i + 2) for i, value in enumerate(x[0][:k]) if value in x[1]]) for x in
           zip(predictions['predictions'], predictions['ground_truth'])]

    IDCG = predictions.apply(lambda x: sum([1 / np.log2(i + 2) for i in range(min(len(x['ground_truth']), k))]),
                             axis=1)
    NDCG = DCG / IDCG
    print(f"Evaluation done in {datetime.datetime.now() - now}")
    return NDCG


def Recall_at_k_batch(predictions, ground_truth, k=100, limit=1000):
    now = datetime.datetime.now()
    predictions = predictions.head(limit)
    predictions['ground_truth'] = ground_truth.head(limit).tolist()
    TPs = [len([value for value in x[0][:k] if value in x[1]]) for x in
           zip(predictions['predictions'], predictions['ground_truth'])]

    # total = predictions.apply(lambda x: min(k, len(x['ground_truth'])), axis=1)
    total = predictions.apply(lambda x: len(x['ground_truth']), axis=1)
    recalls = TPs / total
    print(f"Evaluation done in {datetime.datetime.now() - now}")
    return recalls

