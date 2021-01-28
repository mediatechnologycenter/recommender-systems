import pandas as pd
import os
import sys
import json
import numpy as np
sys.path.append(os.getcwd())

from idea1 import nn_train, prediction, preprocessing, load_embedding
from preprocessing import load_data_cv, get_metadata
from evaluation import evaluate
from helper import restrict_articles_to_timeframe

import click

@click.command()
@click.option('--cut', default=40, help='Only consider users with a least 40 clicks')
@click.option('--high_cut', default=1000,help='only consider last 1000 clicks per user')
@click.option('--seed', default=1,help='random seed for reproducability')
@click.option('--name', default='client_a_both',help='client_a/client_b for solo run, client_a_both/client_b_both for merged run')
def run(cut,high_cut,seed,name):
    """
    Merges the data and train on individual or merged data. We take the same data for client a and client b.
    We output the metrics    for three usergroups: all users, low click users, high click users.

    """
    N = 300  # number of predictions
    limit = 10000000  # number of samples to look at
    first_time = True
    folder_a = 'processed'
    folder_b = 'processed'
    # load data
    client_a_user_item_train, client_a_user_item_test, client_a_user_item_validation = load_data_cv(folder=folder_a,cut=cut,high_cut=high_cut, seed=seed)
    client_a_metadata = get_metadata(folder=folder_a)
    client_a_user_item_test = client_a_user_item_test[client_a_user_item_test.index.isin(client_a_user_item_train.index)]

    client_b_user_item_train, client_b_user_item_test, client_b_user_item_validation = load_data_cv(folder=folder_b,cut=cut,high_cut=high_cut, seed=seed)
    client_b_metadata = get_metadata(folder=folder_b)
    client_b_user_item_test = client_b_user_item_test[client_b_user_item_test.index.isin(client_b_user_item_train.index)]

    # add suffix to article/user_id. Only needed if ids overlap
    client_a_user_item_train=client_a_user_item_train.apply(lambda x: [str(article_id)+"_a" for article_id in x])
    client_a_user_item_test=client_a_user_item_test.apply(lambda x: [str(article_id)+"_a" for article_id in x])
    client_a_metadata['resource_id']=client_a_metadata['resource_id'].astype(str)+"_a"
    client_a_user_item_train.index=client_a_user_item_train.index.astype(str)+"_a"
    client_a_user_item_test.index=client_a_user_item_test.index.astype(str)+"_a"
    
    client_b_user_item_train=client_b_user_item_train.apply(lambda x: [str(article_id)+"_b" for article_id in x])
    client_b_user_item_test=client_b_user_item_test.apply(lambda x: [str(article_id)+"_b" for article_id in x])
    client_b_metadata['resource_id']=client_b_metadata['resource_id'].astype(str)+"_b"
    client_b_user_item_train.index=client_b_user_item_train.index.astype(str)+"_b"
    client_b_user_item_test.index=client_b_user_item_test.index.astype(str)+"_b"


    model_params = {
                    'lr':0.0001,
                    'batch_size': 400,
                    'epochs': 30,
                    'alpha': 1,
                    'layers': [1024, 1024, 16],
                    'dropout':0.5,
                    'reg':0.0001,
                    'interval':1,
                    'checkpoint_interval':1,
                    'eval_batch':False,
                    "normalize":0,
                    "take_target_out":False,
                    "early_stopping":4,
                    "stop_on_metric": True,
                    "loss":"BPR",
                    "optimizer":"ADAM",
                    "workers":3,
                    }
    print(f"Data loaded")

    folder_a=f"{folder_a}/a"
    folder_b=f"{folder_b}/b"
    # embedd data
    if first_time:
        preprocessing(client_a_user_item_train, client_a_metadata, folder=folder_a, model_params=model_params)
        print("embedded")
        preprocessing(client_b_user_item_train, client_b_metadata, folder=folder_b, model_params=model_params)

    # get embedding
    client_b_article_embedding, client_b_user_embedding = load_embedding(folder=folder_b)
    client_a_article_embedding, client_a_user_embedding = load_embedding(folder=folder_a)

    #define data to train on
    both_user_item_train = pd.concat([client_b_user_item_train, client_a_user_item_train])
    both_user_item_test = pd.concat([client_b_user_item_test, client_a_user_item_test])
    both_user_embedding = pd.concat([client_b_user_embedding, client_a_user_embedding])
    both_article_embedding = pd.concat([client_b_article_embedding, client_a_article_embedding])

    if name=='client_a':
        user_item_train,user_item_test,user_embedding,article_embedding_train,new=client_a_user_item_train, client_a_user_item_test, client_a_user_embedding, client_a_article_embedding, True
        article_embedding = article_embedding_train

    if name=='client_b':
        user_item_train,user_item_test,user_embedding,article_embedding_train,new=client_b_user_item_train, client_b_user_item_test, client_b_user_embedding, client_b_article_embedding, True
        article_embedding = article_embedding_train


    if name=='client_a_both':
        user_item_train,user_item_test,user_embedding,article_embedding_train,new=both_user_item_train, client_a_user_item_test, both_user_embedding, both_article_embedding, True
        article_embedding=client_a_article_embedding


    if name=='client_b_both':
        user_item_train,user_item_test,user_embedding,article_embedding_train,new=both_user_item_train, client_b_user_item_test, both_user_embedding, both_article_embedding, True
        article_embedding=client_b_article_embedding



    #train

    model, history = nn_train(user_item_train,user_item_test.sample(frac=0.3, random_state=seed+1),
                     user_embedding=user_embedding,
                     article_embedding=[article_embedding_train,article_embedding], new=new,
                     model_params=model_params,
                     model_path=f'idea1_models/{name}_{cut}_{high_cut}_{seed}',
                     last_x_articles=high_cut)


    user_item_test_sample=user_item_test
    pred,pred_raw = prediction(model, user_embedding.loc[user_item_test_sample.index], article_embedding,
                      user_item_train.loc[user_item_test_sample.index], N, model_params=model_params)
    pred = pred[pred.index.isin(user_item_test_sample.index)]
    idea1 = evaluate(pred.sort_index(), user_item_test_sample.loc[pred.index].sort_index(), limit=limit,experiment_name=f'result_{name}_{new}_{cut}_{high_cut}_{seed}.results')


    user_item_test_sample_low=user_item_test.loc[user_item_test.str.len().sort_values().head(int(len(user_item_test)/4)).index]
    pred,pred_raw = prediction(model, user_embedding.loc[user_item_test_sample_low.index], article_embedding,
                      user_item_train.loc[user_item_test_sample_low.index], N, model_params=model_params)
    pred = pred[pred.index.isin(user_item_test_sample_low.index)]
    idea1 = evaluate(pred.sort_index(), user_item_test_sample_low.loc[pred.index].sort_index(), limit=limit,experiment_name=f'result_{name}_{new}_{cut}_{high_cut}_{seed}_low_click_users.results')


    user_item_test_sample_high=user_item_test.loc[user_item_test.str.len().sort_values().tail(int(len(user_item_test)/4)).index]
    pred,pred_raw = prediction(model, user_embedding.loc[user_item_test_sample_high.index], article_embedding,
                      user_item_train.loc[user_item_test_sample_high.index], N, model_params=model_params)
    pred = pred[pred.index.isin(user_item_test_sample_high.index)]
    idea1 = evaluate(pred.sort_index(), user_item_test_sample_high.loc[pred.index].sort_index(), limit=limit,experiment_name=f'result_{name}_{new}_{cut}_{high_cut}_{seed}_high_click_users.results')


if __name__ == "__main__":
    run()
