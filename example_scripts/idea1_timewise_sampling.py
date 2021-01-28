import os
import sys

sys.path.append(os.getcwd())

from idea1 import nn_train, prediction, preprocessing, load_embedding,load_neg_sampling,get_timewise_neg_samples

from evaluation import evaluate

from preprocessing import load_data, get_metadata, load_data_vertical

if __name__ == "__main__":

    #### timewise sampling
    N = 50  # number of predictions
    limit = 5000  # number of samples to look at
    first_time = True
    folder = os.getenv('PARTNER_A_OUTPUT_FOLDER', 'processed')
    model_params = {'lr': 0.0001,
                    'batch_size': 400,
                    'epochs': 50,
                    'alpha': 1,
                    'layers': [128, 32, 8],
                    'dropout': 0.5,
                    'early_stopping':3,
                    'stop_on_metric':True,
                    'random_sampling':False,
                    "folder": folder,
                    }

    metadata = get_metadata(folder=folder, usecols=['resource_id', 'text'])  # slow
    user_item_train, user_item_test, user_item_validation = load_data_vertical(folder=folder)
    print(f"Data loaded")

    user_item_train['time']=user_item_train['ts']
    user_item_test['time']=user_item_test['ts']
    user_item_train = user_item_train.sort_values('time').drop(columns=['event_ix', 'article_id', 'count', 'percentile', 'ts'], errors='ignore')
    user_item_test = user_item_test.sort_values('time').drop(columns=['event_ix', 'article_id', 'count', 'percentile', 'ts'], errors='ignore')
    user_item_train=user_item_train.head(limit)
    user_item_test = user_item_test.head(limit)

    if first_time:
        get_timewise_neg_samples(user_item_train,user_item_test,folder,split_size=1000000)


    user_item_train,user_item_test=load_neg_sampling(folder)


    if first_time:
        preprocessing(user_item_train, metadata, folder=folder, model_params=model_params)
    article_embedding, user_embedding = load_embedding(folder=folder)
    print(f"Embedding loaded")



    test = user_item_test.groupby('user_ix').resource_id.apply(lambda x: list(x))
    if not os.path.exists('idea1_models'):
        os.makedirs('idea1_models')

    model, history = nn_train(user_item_train, user_item_test[user_item_test['user_ix'].isin(test.index)],
                              user_embedding=user_embedding,
                              article_embedding=article_embedding, new=True,
                              model_params=model_params,
                              model_path='idea1_models/timewise_sampling',
                              last_x_articles=1000)

    test=test.head(10000)
    train_grouped = user_item_train[user_item_train['user_ix'].isin(test.index)].groupby(
        'user_ix').resource_id.apply(lambda x: list(x))
    pred,raw = prediction(model, user_embedding.loc[test.index], article_embedding, train_grouped,
                          N,model_params=model_params)
    pred = pred[pred.index.isin(test.index)]
    idea1 = evaluate(pred.sort_index(), test.sort_index(),
                     experiment_name='idea1_timewise_sampling.results', limit=limit)


