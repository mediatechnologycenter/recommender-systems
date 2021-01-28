import os
import sys

sys.path.append(os.getcwd())

from idea1 import nn_train, prediction, preprocessing, load_embedding,load_neg_sampling,get_timewise_neg_samples

from preprocessing import load_data, get_metadata
from evaluation import evaluate

if __name__ == "__main__":


    N = 50  # number of predictions
    limit = 5000  # number of samples to look at
    first_time = True
    folder = os.getenv('PARTNER_A_OUTPUT_FOLDER', 'processed')

    user_item_train, user_item_test, user_item_validation = load_data(folder=folder, cut=40)
    metadata = get_metadata(folder=folder, usecols=['resource_id', 'text'])  # slow
    user_item_train = user_item_train.head(limit)
    user_item_test = user_item_test.head(limit)

    print(f"Data loaded")
    model_params = {'lr': 0.0001,
                    'batch_size': 400,
                    'epochs': 4,
                    'alpha': 1,
                    'layers': [[128, 32, 8],[8]],
                    'dropout': 0.5,
                    }
    if first_time:
        preprocessing(user_item_train, metadata, folder=folder, model_params=model_params)
    article_embedding, user_embedding = load_embedding(folder=folder)
    print(f"Embedding loaded")



    if not os.path.exists('idea1_models'):
        os.makedirs('idea1_models')
    model, history = nn_train(user_item_train.head(limit), user_item_test.head(1000),
                              user_embedding=user_embedding,
                              article_embedding=article_embedding, new=True,
                              model_params=model_params,
                              model_path='idea1_models/modelname',
                              last_x_articles=1000)
    pred,raw = prediction(model, user_embedding.loc[user_item_train.head(limit).index],
                      article_embedding, user_item_train.head(limit), N,model_params=model_params)
    pred = pred[pred.index.isin(user_item_test.index)]
    idea1 = evaluate(pred.sort_index(), user_item_test.loc[pred.index].sort_index(),
                     experiment_name='idea2.results', limit=limit)
