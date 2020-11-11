import pandas as pd
import pickle
import numpy as np
import os
import datetime
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Dense, Input, Embedding, concatenate, Flatten,Dropout
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
import tensorflow.keras.backend as K
import math

"""
Implementation of our new approach. As an overview: we embed the articles with some embedding (Sentence Bert).
We calculate user-vector by taking the average over all article-vectors from articles the user read.
We feed the user-vector together with a article-vector that the user read (pos_sample) and one she/he did not 
read (neg_sample) into a neural network and use pairwise loss to train the model weights.
"""

def preprocessing(grouped_train,metadata, folder):
    """
    Creates an embedding of the articles and users and saves it
    """
    print("preprocessing...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    embeddings = embedder.encode(list(metadata['text']), convert_to_tensor=False)

    article_vectors = pd.DataFrame()
    for i, article_id in enumerate(metadata['resource_id']):
        article_vectors[article_id] = embeddings[i]
    lookup = article_vectors.T
    pickle.dump(lookup, open(f"{folder}/article_embedding.pkl", "wb+"))

    print("Embedded Articles...")

    ##### get user_vector
    from pandarallel import pandarallel
    pandarallel.initialize(use_memory_fs=False)
    user_vectors = grouped_train.parallel_apply(lambda x: lookup.loc[x, :].mean())
    # user_vectors = grouped_train.apply(lambda x: lookup.loc[x, :].mean())
    user_vectors.columns = [str(i) for i in user_vectors]
    user_vectors.to_parquet(f"{folder}/user_embedding.pq")

    print("Embedded Users...")

def load_embedding(folder):
    article_embedding = pd.read_pickle(f"{folder}/article_embedding.pkl")
    user_embedding = pd.read_parquet(f"{folder}/user_embedding.pq")
    return article_embedding, user_embedding

def nn_train(train,user_embedding,article_embedding, model_params, model_path,new=False, last_x_articles=1000 ):
    """
    Trains the model and returns it. 
    :param train: pd.Series of training data to use. Index is the UserID, value is a list of ArticleIDs the user read.
    :param user_embedding: Embedding Dataframe of the Users. Index is the UserID, values is the embeddingvector
    :param article_embedding: Embedding Dataframe of the Articles. Index is the ArticleID, values is the embeddingvector
    :param model_params: Hyperparameters for the model
    :param model_path: Path to save the model
    :param new: If set to False it simply loads the model saved under model_path
    :param last_x_articles: Restrict to the last last_x_articles the each user read.
    :return: Returns the model and the trainig history
    """

    now = datetime.datetime.now()
    if new:

        val = train.str[-last_x_articles:].head(int(len(train)*0.1))
        train = train.str[-last_x_articles:].tail(len(train)-int(len(train)*0.1))
        model,history = nn_model(article_embedding, user_embedding, train,val, model_params,model_path=model_path)
        # model=None
    else:
        model=   tf.keras.models.load_model(f'{model_path}')
        history=""
    return model,history

def explode_data_and_add_original(grouped):
    """
    Helper function for upsamping
    """
    grouped_upsampling = pd.DataFrame(grouped.explode())
    grouped_upsampling.columns = ['pos_sample']
    grouped_upsampling['pos_samples'] = grouped
    grouped_upsampling = grouped_upsampling.reset_index()
    return grouped_upsampling
class data_generator(tf.keras.utils.Sequence):
    """
    Class to create and feed the batches. For each user-item sample in the training set samples neg_samples_per_user
    negative samples i.e. articles he did not read. Returns batches of size batch_size where each sample is:
    ((uservector, pos_sample_article_vector),(uservector, neg_sample_article_vector))
    """

    def __init__(self, grouped_exploded, uservector, lookup, batch_size, neg_samples_per_user=10):
        self.grouped_upsampling = grouped_exploded.sample(frac=1)#explode_data_and_add_original(grouped_exploded).sample(frac=1)
        self.neg_samples_per_user = neg_samples_per_user
        self.batch_size = batch_size
        self.new_batch_size = int(self.batch_size * 0.8)
        self.lookup = lookup
        self.uservector = uservector
        self.len=math.ceil(len(self.grouped_upsampling) / self.batch_size) * self.neg_samples_per_user
        self.seen=0

    def __len__(self):
        return self.len

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.seen>=self.len:
            self.grouped_upsampling = self.grouped_upsampling.sample(frac=1)
            self.seen=0
            print('shuffeled')
    def __getitem__(self, idx):
        if idx==1:
            print('test')
        self.seen=self.seen+1
        try:
            i = idx % math.ceil(len(self.grouped_upsampling) / self.batch_size)
            batch = self.grouped_upsampling.iloc[i * self.batch_size:(i + 1) * self.batch_size, :]
            # batch = pd.concat([batch] * self.neg_samples_per_user)

            for i in range(9):
                batch['neg_sample'] = list(pd.Series(self.lookup.index).sample(n=len(batch), replace=True))
                batch_new = batch[[x[1] not in x[0] for x in zip(batch['pos_samples'], batch['neg_sample'])]][
                    ['user_ix', 'pos_sample', 'neg_sample']]

                if len(batch_new) > self.new_batch_size:
                    break
            if i == 8:
                batch_new = batch_new.sample(n=self.new_batch_size, replace=True)
                print("hmmmm")
            else:
                batch_new = batch_new.sample(n=self.new_batch_size)
            batch_pos, batch_neg = self.create_batch(batch_new)
        except KeyError:
            print("hopla")
            return self.__getitem__(idx+1)
        assert not np.any(np.isnan(batch_pos))
        assert not np.any(np.isnan(batch_neg))
        return ((batch_pos, batch_neg), batch_pos)

    def create_batch(self, batch):
        users = batch['user_ix'].values
        pos_samples = batch['pos_sample'].values
        neg_samples = list(batch['neg_sample'])

        return np.concatenate((self.uservector.loc[users, :].values, self.lookup.loc[pos_samples, :].values),
                              axis=1), \
               np.concatenate((self.uservector.loc[users, :].values, self.lookup.loc[neg_samples, :].values),
                              axis=1)

def nn_model(lookup, uservector, train,val, model_params,model_path):
    """
    Defines the model and trains it
    """
    lr = model_params['lr']
    batch_size = model_params['batch_size']
    epochs = model_params['epochs']
    neg_samples_per_user = model_params['neg_samples_per_user']
    alpha = model_params['alpha']
    layers = model_params['layers']
    dropout=model_params.get('dropout',0.5)
    reg=model_params.get('reg',True)
    steps = model_params.get('steps',int(train.str.len().sum() / batch_size) * neg_samples_per_user)

    def custom_loss(y_pred_pos, y_pred_neg, alpha=1):
        alpha = K.constant(alpha)
        pointwise_loss = -K.log(y_pred_pos + 1e-07)
        pairwise_loss = -K.log(K.sigmoid(y_pred_pos - y_pred_neg)+ 1e-07)
        loss = alpha * pairwise_loss + (1 - alpha) * pointwise_loss

        return loss

    embedding_size = lookup.shape[1] * 2

    with tf.device('/gpu:0'):
        texts_pos = Input(shape=(embedding_size,))
        texts_neg = Input(shape=(embedding_size,))
        if reg:
            sigmoid = Dense(1, activation='sigmoid',kernel_constraint=unit_norm(), bias_constraint=unit_norm())
        else:
            sigmoid = Dense(1, activation='sigmoid')#,kernel_constraint=unit_norm(), bias_constraint=unit_norm())

        n_out=texts_neg
        p_out=texts_pos

        for layer_size in layers:
            drop = Dropout(dropout)
            if reg:
                layer = Dense(layer_size, activation=tf.nn.relu,kernel_constraint=unit_norm(), bias_constraint=unit_norm())
            else:
                layer = Dense(layer_size, activation=tf.nn.relu)#,kernel_constraint=unit_norm(), bias_constraint=unit_norm())
            n_out=layer(drop(n_out))
            p_out=layer(drop(p_out))

        y_pos = sigmoid(p_out)
        y_neg = sigmoid(n_out)
        model = Model(inputs=[texts_pos, texts_neg], outputs=[y_pos, y_neg])

        loss = custom_loss(y_pos, y_neg, alpha)

        # Add loss to model
        model.add_loss(loss)

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), )
        model.summary()

        print(int(train.str.len().sum() / batch_size) * neg_samples_per_user)

        seq_train = data_generator(explode_data_and_add_original(train), lookup=lookup, uservector=uservector,
                             batch_size=batch_size,
                             neg_samples_per_user=neg_samples_per_user)
        seq_validation = data_generator(explode_data_and_add_original(val), lookup=lookup, uservector=uservector,
                             batch_size=batch_size,
                             neg_samples_per_user=1)
        history = model.fit(seq_train,validation_data=seq_validation,
                            epochs=epochs, steps_per_epoch=steps, verbose=1, max_queue_size=1000, shuffle=False,
                            use_multiprocessing=True, workers=8)


        model.save(f'{model_path}')
        del seq_train
        del seq_validation
    return model,history

class test_data_generator(tf.keras.utils.Sequence):

        def __init__(self, uservector, lookup):
            self.lookup = lookup
            self.uservector = uservector

        def __len__(self):
            return len(self.uservector)

        def __getitem__(self, idx):
            user = self.uservector.iloc[idx, :]
            sample = np.concatenate((np.tile(user, (len(self.lookup), 1)), self.lookup.values), axis=1)
            return ((sample, sample), sample)

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            pass

def prediction(model, users, lookup,user_item_train, N,gen=None):
    """
    Creates prediction for each article for each user in the testset. Filters out already read articles. Sorts the
    predicted score and returns the top N articles.
    """
    if gen is None:
        gen = test_data_generator(users, lookup)
    pred = model.predict(gen, use_multiprocessing=False, verbose=1, workers=15)
    pred = pd.DataFrame(pred[0])
    pred['article'] = lookup.index.fillna(-1).to_list() * len(users)
    pred.index = np.repeat(users.index, repeats=len(lookup))
    pred['read_articles'] = user_item_train
    isnull = pred['read_articles'].isnull()
    if isnull.sum() > 0:
        pred.loc[isnull, 'read_articles'] = [[[]] * isnull.sum()]
    pred = pred[[x[1] not in x[0] for x in zip(pred['read_articles'], pred['article'])]]
    pred = pred.reset_index()
    pred = pred.sort_values(0, ascending=False).groupby('user_ix').head(N).reset_index().sort_values(0)
    pred = pred.groupby('user_ix').apply(lambda x: list(x['article']))
    pred = pd.DataFrame(pred, columns=['predictions'])
    return pred



if __name__ == "__main__":
    from preprocessing import load_data,get_metadata
    from evaluation import evaluate
    from helper import restrict_articles_to_timeframe

    N = 50  # number of predictions
    limit = 5000  # number of samples to look at
    first_time = True
    folder = os.getenv('PARTNER_A_OUTPUT_FOLDER','processed')

    user_item_train, user_item_test, user_item_validation = load_data(folder=folder, cut=40)
    metadata = get_metadata(folder=folder, usecols=['resource_id', 'text'])  # slow
    user_item_train = user_item_train.head(limit)
    user_item_test = user_item_test.head(limit)

    # Partner A can remove this
    # user_item_train, metadata = restrict_articles_to_timeframe(user_item_train, metadata,
    #                                                                    start_date=datetime.date(2020, 1, 1))
    # user_item_test, metadata = restrict_articles_to_timeframe(user_item_test, metadata,
    #                                                                    start_date=datetime.date(2020, 1, 1))

    print(f"Data loaded")

    if first_time:
        preprocessing(user_item_train, metadata, folder=folder)
    article_embedding, user_embedding = load_embedding(folder=folder)
    print(f"Embedding loaded")

    model_params = {'lr': 0.0001,
                    'batch_size': 400,
                    'epochs': 4,
                    'neg_samples_per_user': 1,
                    'alpha': 1,
                    'layers': [128, 32, 8],
                    'dropout': 0.5,
                    }

    if not os.path.exists('idea1_models'):
        os.makedirs('idea1_models')
    model, history = nn_train(user_item_train.head(limit),
                                  user_embedding=user_embedding,
                                  article_embedding=article_embedding, new=True,
                                  model_params=model_params,
                                  model_path='idea1_models/modelname',
                                  last_x_articles=1000)

    pred = prediction(model, user_embedding.loc[user_item_train.head(limit).index],
                          article_embedding, user_item_train.head(limit), N)
    pred = pred[pred.index.isin(user_item_test.index)]
    idea1 = evaluate(pred, user_item_test.loc[pred.index],
                     experiment_name='resultsname', limit=limit)
