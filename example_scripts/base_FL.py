import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import sys

sys.path.append(os.getcwd())
from idea1 import nn_train, prediction, preprocessing, load_embedding
from preprocessing import load_data_cv, get_metadata
from evaluation import evaluate

import click


def get_weight_updates(local_weights, global_weights):
    updates = np.array(local_weights) - np.array(global_weights)

    return updates


def get_average_updates(list_updates, global_weights, rate_updates=1):
    updates = np.mean(list_updates, axis=0)
    global_weights = global_weights + rate_updates * updates

    return global_weights



@click.command()
@click.option('--cut', default=40, help='Only consider users with a least 40 clicks')
@click.option('--high_cut', default=1000,help='only consider last 1000 clicks per user')
@click.option('--seed', default=1,help='random seed for reproducability')
@click.option('--epsilon', default=0.0, help='amount of noise to add in terms of differential privacy')
@click.option('--lr', default=0.00001,help='learning rate')
def run(cut, high_cut, seed, epsilon, lr):
    """
    Trains a idea1 model in a federated way. We take the same data for client a and client b. We output the metrics
    for three usergroups: all users, low click users, high click users.

    """
    N = 300  # number of predictions
    limit = 50000000  # number of samples to look at
    first_time = True
    folder_a = 'processed'
    folder_b = 'processed'
    client_num=0 #which client to evaluate on
    model_params = {
        'lr': 0.0001,
        'batch_size': 400,
        'epochs': 1,

        'alpha': 1,
        'layers': [128, 16],
        "stop_on_metric": True,
        'dropout': 0.5,
        'reg': 0.0001,
        'interval': 1,
        'checkpoint_interval': 1,
        'eval_batch': False,
        "normalize": 0,
        "take_target_out": False,
        "early_stopping": 4,
        "loss": "BPR",
        "optimizer": "ADAM",
        "workers": 1,
        "epsilon": epsilon,
    'rounds': 50
    }

    #load data
    client_a_user_item_train, client_a_user_item_test, client_a_user_item_validation = load_data_cv(folder=folder_b,
                                                                                                    cut=cut,
                                                                                                    high_cut=high_cut,
                                                                                                    seed=seed)
    client_a_user_item_train = client_a_user_item_train.head(limit)
    client_a_user_item_test = client_a_user_item_test[
        client_a_user_item_test.index.isin(client_a_user_item_train.index)]
    client_a_group_metadata = get_metadata(folder=folder_b)

    client_b_user_item_train, client_b_user_item_test, client_b_user_item_validation = load_data_cv(folder=folder_a,
                                                                                                    cut=cut,
                                                                                                    high_cut=high_cut,
                                                                                                    seed=seed)
    client_b_user_item_train = client_b_user_item_train.head(limit)
    client_b_metadata = get_metadata(folder=folder_a)
    client_b_user_item_test = client_b_user_item_test[
        client_b_user_item_test.index.isin(client_b_user_item_train.index)]

    print(f"Data loaded")
    # embedd data
    if first_time:
        preprocessing(client_a_user_item_train, client_a_group_metadata, folder=folder_b, model_params=model_params)
        print("embedded")
        preprocessing(client_b_user_item_train, client_b_metadata, folder=folder_a, model_params=model_params)
    client_b_article_embedding, client_b_user_embedding = load_embedding(folder=folder_a)
    client_a_article_embedding, client_a_user_embedding = load_embedding(folder=folder_b)

    #dict for federated learning
    clients = [{"name": "b",
                "user_item_train": client_b_user_item_train,

                "user_item_test": client_b_user_item_test,
                "user_embedding": client_b_user_embedding,
                "article_embedding": client_b_article_embedding,

                },
               {"name": "a",
                "user_item_train": client_a_user_item_train,

                "user_item_test": client_a_user_item_test,
                "user_embedding": client_a_user_embedding,
                "article_embedding": client_a_article_embedding,

                }
               ]

    #prepare inital global model
    model_params['train'] = False
    params = f"{clients[client_num]['name']}_{cut}_{high_cut}_{seed}_{epsilon}_{lr}"

    global_model, history = nn_train(clients[0]['user_item_train'],
                                     clients[0]['user_item_test'].sample(frac=0.3, random_state=seed + 1),
                                     user_embedding=clients[0]['user_embedding'],
                                     article_embedding=[clients[0]['article_embedding'],
                                                        clients[0]['article_embedding']],
                                     model_params=model_params.copy(),
                                     new=True,
                                     model_path=f'idea1_models/fl/fl_client_b_client_a_global_model_{params}',
                                     last_x_articles=high_cut)

    model_params['train'] = True
    for client in clients:
        if not os.path.exists(f'results/idea1_models/fl/fl_{client["name"]}_local_model_{params}'):
            os.makedirs(f'results/idea1_models/fl/fl_{client["name"]}_local_model_{params}')
        open(f'results/idea1_models/fl/fl_{client["name"]}_local_model_{params}/metrics', "w").close()

    # commence global training loop
    for comm_round in range(model_params['rounds']):
        model_params['round'] = comm_round
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()
        print(comm_round)
        # initial list to collect local model weights
        local_weight_updates_list = list()

        # loop through each client and create and train new local model
        for client in clients:
            local_model, history = nn_train(client['user_item_train'],
                                            client['user_item_test'].sample(frac=0.3, random_state=seed + 1),
                                            user_embedding=client['user_embedding'],
                                            article_embedding=[client['article_embedding'],
                                                               client['article_embedding']],
                                            model_params=model_params.copy(),
                                            new=False,
                                            retrain=True,
                                            new_model_path=f'idea1_models/fl/fl_{client["name"]}_local_model_{params}',

                                            model_path=f'idea1_models/fl/fl_client_b_client_a_global_model_{params}',
                                            last_x_articles=high_cut)

            updates = get_weight_updates(local_model.get_weights(), global_weights)
            local_weight_updates_list.append(updates)

            K.clear_session()


        # stop if metrics do not increase for the target client
        df = pd.read_csv(f'results/idea1_models/fl/fl_{clients[client_num]["name"]}_local_model_{params}/metrics')
        df.columns = ['name', f'metric', f'value', f'std']
        dfgroup = df.groupby('metric').tail(10)
        dfgroup['epoch'] = (dfgroup.index / 7).astype(int)

        ndcg100 = dfgroup[dfgroup['metric'].str.contains('Recall@10')][['value', 'epoch']]
        top = ndcg100.sort_values('value').tail(1)
        stop2 = len(ndcg100.tail(model_params["early_stopping"])[ndcg100.tail(model_params["early_stopping"])['value'] > top.iloc[0, 0]]) == 0
        print(ndcg100.tail(model_params["early_stopping"])[ndcg100.tail(model_params["early_stopping"])['value'] > top.iloc[0, 0]])

        ndcg100 = dfgroup[dfgroup['metric'].str.contains('NDCG@100')][['value', 'epoch']]
        top = ndcg100.sort_values('value').tail(1)
        print(ndcg100.tail(model_params["early_stopping"])[ndcg100.tail(model_params["early_stopping"])['value'] > top.iloc[0, 0]])
        stop1 = len(ndcg100.tail(model_params["early_stopping"])[ndcg100.tail(model_params["early_stopping"])['value'] > top.iloc[0, 0]]) == 0

        if stop1 and stop2 and comm_round!=0:
            epoch = top['epoch'].iloc[0]
            print(epoch)
            model = tf.keras.models.load_model(
                f'idea1_models/fl/fl_{clients[client_num]["name"]}_local_model_{params}_epochs/{epoch}.h5', )
            model.save(f'idea1_models/fl/fl_{clients[client_num]["name"]}_local_model_{params}.h5', )
            break


        # update global model
        rate_updates = 1
        average_weights = get_average_updates(local_weight_updates_list, global_weights, rate_updates)
        global_model.set_weights(average_weights)
        if not os.path.exists(f'idea1_models/fl/fl_client_b_client_a_global_model_{params}/epochs/'):
            os.makedirs(f'idea1_models/fl/fl_client_b_client_a_global_model_{params}/epochs/')
        global_model.save(f'idea1_models/fl/fl_client_b_client_a_global_model_{params}.h5')
        global_model.save(f'idea1_models/fl/fl_client_b_client_a_global_model_{params}/epochs/{comm_round}.h5')


    # evaluate
    model_params['train'] = False
    model, history = nn_train(clients[client_num]['user_item_train'],
                              clients[client_num]['user_item_test'].sample(frac=0.3, random_state=seed + 1),
                              user_embedding=clients[client_num]['user_embedding'],
                              article_embedding=[clients[client_num]['article_embedding'],
                                                 clients[client_num]['article_embedding']],
                              model_params=model_params,
                              new=False,
                              model_path=f'idea1_models/fl/fl_{clients[client_num]["name"]}_local_model_{params}',
                              last_x_articles=high_cut)
    user_item_test = clients[client_num]['user_item_test']
    user_embedding = clients[client_num]['user_embedding']
    article_embedding = clients[client_num]['article_embedding']
    user_item_train = clients[client_num]['user_item_train']
    user_item_test_sample = user_item_test


    pred, pred_raw = prediction(model, user_embedding.loc[user_item_test_sample.index], article_embedding,
                                user_item_train.loc[user_item_test_sample.index], N, model_params=model_params)
    pred = pred[pred.index.isin(user_item_test_sample.index)]
    idea1 = evaluate(pred.sort_index(), user_item_test_sample.loc[pred.index].sort_index(), limit=limit,
                     experiment_name=f'result_{params}_all_users.results')

    user_item_test_sample_low = user_item_test.loc[user_item_test.str.len().sort_values().head(int(len(user_item_test)/4)).index]
    pred, pred_raw = prediction(model, user_embedding.loc[user_item_test_sample_low.index], article_embedding,
                                user_item_train.loc[user_item_test_sample_low.index], N, model_params=model_params)
    pred = pred[pred.index.isin(user_item_test_sample_low.index)]
    idea1 = evaluate(pred.sort_index(), user_item_test_sample_low.loc[pred.index].sort_index(), limit=limit,
                     experiment_name=f'result_{params}_low_click_users.results')

    user_item_test_sample_high = user_item_test.loc[user_item_test.str.len().sort_values().tail(int(len(user_item_test)/4)).index]
    pred, pred_raw = prediction(model, user_embedding.loc[user_item_test_sample_high.index], article_embedding,
                                user_item_train.loc[user_item_test_sample_high.index], N, model_params=model_params)
    pred = pred[pred.index.isin(user_item_test_sample_high.index)]
    idea1 = evaluate(pred.sort_index(), user_item_test_sample_high.loc[pred.index].sort_index(), limit=limit,
                     experiment_name=f'result_{params}_high_click_users.results')


import datetime

if __name__ == "__main__":
    run()
