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

import pandas as pd
import gc
import pickle
import numpy as np
import os
import datetime
import random
from evaluation import evaluate
import shelve
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import math
import tensorflow as tf
import os, psutil

process = psutil.Process(os.getpid())
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

"""
Implementation of our new approach. As an overview: we embed the articles with some embedding (Sentence Bert).
We calculate user-vector by taking the average over all article-vectors from articles the user read.
We feed the user-vector together with a article-vector that the user read (pos_sample) and one she/he did not 
read (neg_sample) into a neural network and use pairwise loss to train the model weights.
"""


def preprocessing(grouped_train, metadata, folder, model_params):
    """
    Creates an embedding of the articles and users (by averageing over the read article vectors) and saves it
    """
    if not os.path.exists(f'{folder}'):
        os.makedirs(f'{folder}')
    grouped = model_params.get('random_sampling', True)
    print("Preprocessing Embeddings...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    embeddings = embedder.encode(list(metadata['text']), convert_to_tensor=False)

    article_vectors = pd.DataFrame()
    for i, article_id in enumerate(metadata['resource_id']):
        article_vectors[article_id] = embeddings[i]
    lookup = article_vectors.T
    pickle.dump(lookup, open(f"{folder}/article_embedding.pkl", "wb+"))

    if not grouped:
        grouped_train2 = grouped_train.groupby('user_ix').resource_id.apply(list)
        print("Embedded Articles...")
    else:
        grouped_train2 = grouped_train

    ##### get user_vector
    from pandarallel import pandarallel
    pandarallel.initialize(use_memory_fs=False)
    user_vectors = grouped_train2.parallel_apply(lambda x: lookup.loc[x, :].mean())
    # user_vectors = grouped_train.apply(lambda x: lookup.loc[x, :].mean())
    user_vectors.columns = [str(i) for i in user_vectors]
    user_vectors.to_parquet(f"{folder}/user_embedding.pq")

    print("Embedded Users...")


def get_timewise_neg_samples(user_item_train, user_item_test, folder, split_size=100000, k=5):
    store_samples(user_item_train=user_item_train, folder=folder, filename="user_item_train_neg_sampling",
                  split_size=split_size, k=k)
    store_samples(user_item_train=user_item_test, folder=folder, filename="user_item_test_neg_sampling",
                  split_size=split_size, k=k)


def store_samples(user_item_train, folder, filename, split_size, k):
    user_item_train = user_item_train.sort_values('time')
    user_item_train['k'] = range(len(user_item_train))
    grouped_positives = user_item_train.sort_values('time').groupby('user_ix').resource_id.apply(set)
    grouped_positives.name = 'pos_samples'
    splits = int(len(user_item_train) / split_size)

    myShelvedDict = shelve.open(f"{folder}/{filename}.db", flag='n')
    myShelvedDict.close()

    for i in range(splits + 1):
        print(f"Working on negative samples {i}/{splits}")
        start = datetime.datetime.now()
        tmp = user_item_train.iloc[i * split_size:(i + 1) * split_size, :]

        tmp = pd.merge(tmp, grouped_positives, how='left', left_on='user_ix', right_index=True)

        tmp['res'] = "," + tmp['resource_id'].astype(str)
        tmp['neg_samples'] = ''
        for i in range(-4, 6):
            if i == 0:
                continue
            tmp['neg_samples'] = tmp['neg_samples'] + tmp['res'].shift(i, fill_value='')
        tmp['neg_samples'] = tmp['neg_samples'].str[1:].str.split(',').apply(set)
        tmp['neg_samples'] = tmp['neg_samples'] - tmp['pos_samples']
        tmp_fix = tmp[tmp['neg_samples'].str.len() < k]
        tmp_fix['neg_samples'] = tmp_fix['k'].apply(
            lambda k: set(user_item_train['resource_id'].iloc[
                          max(0, k - 100):(k + 1) + 100]))

        tmp_fix['neg_samples'] = tmp_fix['neg_samples'] - tmp_fix['pos_samples']
        tmp.loc[tmp_fix.index, 'neg_samples'] = tmp_fix['neg_samples']

        tmp_fix = tmp[tmp['neg_samples'].str.len() < k]
        tmp_fix['neg_samples'] = tmp_fix['k'].apply(
            lambda k: set(user_item_train['resource_id'].iloc[
                          max(0, k - 10000):(k + 1) + 10000]))

        tmp_fix['neg_samples'] = tmp_fix['neg_samples'] - tmp_fix['pos_samples']
        tmp.loc[tmp_fix.index, 'neg_samples'] = tmp_fix['neg_samples']

        tmp['neg_samples'] = tmp['neg_samples'].apply(lambda x: random.choices(list(x), k=k))
        tmp.index = tmp['k'].astype(str)

        myShelvedDict = shelve.open(f'{folder}/{filename}.db', flag='wf')
        myShelvedDict.update(tmp['neg_samples'].to_dict())
        myShelvedDict.close()
        del tmp

    user_item_train.to_pickle(f'{folder}/{filename}.pkl')


def load_embedding(folder):
    article_embedding = pd.read_pickle(f"{folder}/article_embedding.pkl")
    user_embedding = pd.read_parquet(f"{folder}/user_embedding.pq")
    return article_embedding, user_embedding


def load_neg_sampling(folder):
    user_item_train = pd.read_pickle(f'{folder}/user_item_train_neg_sampling.pkl')
    user_item_test = pd.read_pickle(f'{folder}/user_item_test_neg_sampling.pkl')
    return user_item_train, user_item_test


def nn_train(train, test, user_embedding, article_embedding, model_params, model_path, new=False, last_x_articles=1000,
             retrain=False, new_model_path=''):
    """
    Creates/loads a model, creates a validation split and train the model
    :param train: pd.Series of training data to use. Index is the UserID, value is a list of ArticleIDs the user read.
    :param user_embedding: Embedding Dataframe of the Users. Index is the UserID, values is the embeddingvector
    :param article_embedding: Embedding Dataframe of the Articles. Index is the ArticleID, values is the embeddingvector
    :param model_params: Hyperparameters for the model
    :param model_path: Path to save the model
    :param new: If set to False it simply loads the model saved under model_path
    :param last_x_articles: Restrict to the last last_x_articles the each user read.
    :param retrain: whether a loaded model should  be retrained (True) or not
    :param new_model_path: optional new model path when we retrain
    :return: Returns the model and the training history
    """
    grouped = model_params.get('random_sampling', True)
    now = datetime.datetime.now()
    model = None
    if not new:
        model = tf.keras.models.load_model(f'{model_path}.h5')

        history = ""
        model_path = new_model_path

    if new or retrain:

        if grouped:
            train = train[train.str.len() < 1000]
            val = train.str[-last_x_articles:].head(int(len(train) * 0.1))
            train = train.str[-last_x_articles:].tail(len(train) - int(len(train) * 0.1))
        else:

            val = train.head(int(len(train) * 0.1))
            train = train.tail(len(train) - int(len(train) * 0.1))
        model, history = nn_model(article_embedding, user_embedding, train, val, test, model_params,
                                  model_path=model_path, model=model)
        # model=None

    return model, history


def nn_model(lookup, uservector, train, val, test, model_params, model_path, model):
    """
    Defines the model and trains it
    """
    if len(lookup) != 2:
        lookup = (lookup, lookup)
    model_params['random_sampling'] = model_params.get('random_sampling', True)
    model_params['random_sampling'] = model_params.get('random_sampling', 1)
    model_params['lr'] = model_params.get('lr', 0.00001)
    model_params['batch_size'] = model_params.get('batch_size', 100)
    model_params['epochs'] = model_params.get('epochs', 50)
    model_params['alpha'] = model_params.get('alpha', 1)
    model_params['layers'] = model_params.get('layers', [1024, 512, 8])
    model_params['dropout'] = model_params.get('dropout', 0.5)
    model_params['dropout_first'] = model_params.get('dropout_first', model_params['dropout'])
    model_params['reg'] = model_params.get('reg', 0)
    model_params['normalize'] = model_params.get('normalize', 0)
    model_params['interval'] = model_params.get('interval', 1)
    model_params['checkpoint_interval'] = model_params.get('checkpoint_interval', 1)
    model_params['early_stopping'] = model_params.get('early_stopping', 0)
    model_params['loss'] = model_params.get('loss', "0")
    model_params['optimizer'] = model_params.get('optimizer', "ADAM")
    model_params['take_target_out'] = model_params.get('take_target_out', False)
    model_params['decay'] = model_params.get('decay', False)
    model_params['stop_on_metric'] = model_params.get('stop_on_metric', False)
    model_params['workers'] = model_params.get('workers', 1)
    model_params['train'] = model_params.get('train', True)
    model_params['round'] = model_params.get('round', False)
    model_params['epsilon'] = model_params.get('epsilon', 0)
    history = ''
    train_grouped = train
    test_ids = test
    neg_samples_train = None
    neg_samples_test = None

    if model_params['random_sampling']:
        model_params['steps'] = model_params.get('steps', int(train.str.len().sum() / model_params['batch_size']))
    else:
        # for timewise/custom sampling we get data in vertical mode. In that case we group the data and load the pre-
        # created negative samples
        train_grouped = pd.concat([train[train['user_ix'].isin(test['user_ix'])],
                                   val[val['user_ix'].isin(test['user_ix'])]]
                                  ).groupby('user_ix').resource_id.apply(list)
        test_ids = test.groupby('user_ix').resource_id.apply(list).head(1000)
        model_params['steps'] = model_params.get('steps', int(len(train) / model_params['batch_size']))

        neg_samples_train = shelve.open(f'{model_params["folder"]}/user_item_train_neg_sampling.db', flag='r')
        neg_samples_test = shelve.open(f'{model_params["folder"]}/user_item_test_neg_sampling.db', flag='r')

    # model creation
    if model:
        pass
    else:
        # idea2
        if type(model_params['layers'][0]) == list:
            embedding_size = lookup[0].shape[1]
            model = idea2_architecture(embedding_size, model_params)

        # Idea1
        else:
            embedding_size = lookup[0].shape[1] * 2
            model = idea1_architecture(embedding_size, model_params)

        if model_params['optimizer'] == "ADAM":
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=model_params['lr']))
        else:
            model.compile(optimizer=tf.keras.optimizers.SGD(lr=model_params['lr']))
        model.summary()

    print(model_params)

    with tf.device('/gpu:0'):
        # prepare data generators
        seq_train = data_generator(train, lookup=lookup[0], uservector=uservector, model_params=model_params,
                                   neg_samples=neg_samples_train)
        tmp = model_params['take_target_out']
        model_params['take_target_out'] = False
        # validation is currently done on parts on the test set as it was better for the early stopping. Validation set
        # is not used.
        seq_validation = data_generator(test, lookup=lookup[0], uservector=uservector, model_params=model_params,
                                        neg_samples=neg_samples_test)
        seq_test = test_data_generator(uservector.loc[test_ids.index], lookup[1], model_params=model_params)
        model_params['take_target_out'] = tmp

        #set callbacks
        callback = [IntervalEvaluation(test=test_ids, seq=seq_test, user_embedding=uservector.loc[test_ids.index],
                                       article_embedding=lookup[1],
                                       user_item_train=train_grouped,
                                       model_params=model_params,
                                       model_path=model_path, seq_validation=seq_validation)]

        if model_params['early_stopping'] and not model_params['stop_on_metric']:
            callback.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, mode='min',
                                                             patience=model_params['early_stopping'],
                                                             restore_best_weights=True))

        if model_params['train']:
            history = model.fit(seq_train, validation_data=seq_validation, callbacks=callback,
                                epochs=model_params['epochs'], steps_per_epoch=model_params['steps'], verbose=1,
                                max_queue_size=100, shuffle=False,
                                use_multiprocessing=True, workers=model_params['workers'])

        model.save(f'{model_path}.h5')
        try:
            neg_samples_test.close()
            neg_samples_train.close()
        except:
            pass
        del seq_train
        del seq_validation
        del seq_test
    return model, history


def custom_loss(y_pred_pos, y_pred_neg, model_params):
    alpha = K.constant(model_params['alpha'])
    pointwise_loss = -K.log(y_pred_pos + 1e-07) - K.log(1 - y_pred_neg + 1e-07)

    if model_params['loss'] == 'TOP':
        pairwise_loss = K.sigmoid(y_pred_neg - y_pred_pos) + K.sigmoid(y_pred_neg * y_pred_neg)

    else:
        pairwise_loss = -K.log(
            K.sigmoid(y_pred_pos - y_pred_neg) + 1e-07)
    loss = alpha * pairwise_loss + (1 - alpha) * pointwise_loss

    return tf.reduce_mean(loss)


def idea2_architecture(embedding_size, model_params):
    user = Input(shape=(embedding_size,))
    texts_pos = Input(shape=(embedding_size,))
    texts_neg = Input(shape=(embedding_size,))

    drop = Dropout(model_params['dropout_first'])
    n_out = drop(texts_neg)
    p_out = drop(texts_pos)
    u_out = drop(user)

    for layer_size in model_params['layers'][0]:
        drop_article = Dropout(model_params['dropout'])
        if model_params['reg'] != 0:
            layer_article = Dense(layer_size, activation=tf.nn.relu,
                                  kernel_regularizer=regularizers.l2(model_params['reg']))
        else:
            layer_article = Dense(layer_size,
                                  activation=tf.nn.relu)  # ,kernel_constraint=unit_norm(), bias_constraint=unit_norm())

        drop_user = Dropout(model_params['dropout'])
        if model_params['reg'] != 0:
            layer_user = Dense(layer_size, activation=tf.nn.relu,
                               kernel_regularizer=regularizers.l2(model_params['reg']))
        else:
            layer_user = Dense(layer_size,
                               activation=tf.nn.relu)  # ,kernel_constraint=unit_norm(), bias_constraint=unit_norm())

        n_out = drop_article(layer_article(n_out))
        p_out = drop_article(layer_article(p_out))

        u_out = drop_user(layer_user(u_out))

    concat = Concatenate()
    y_pos = concat([u_out, p_out])
    y_neg = concat([u_out, n_out])

    for layer_size in model_params['layers'][1]:
        drop_con = Dropout(model_params['dropout'])
        if model_params['reg'] != 0:
            layer_con = Dense(layer_size, activation=tf.nn.relu,
                              kernel_regularizer=regularizers.l2(model_params['reg']))
        else:
            layer_con = Dense(layer_size,
                              activation=tf.nn.relu)  # ,kernel_constraint=unit_norm(), bias_constraint=unit_norm())

        y_pos = drop_con(layer_con(y_pos))
        y_neg = drop_con(layer_con(y_neg))

    sigm = Dense(1, activation='sigmoid')

    y_pos = sigm(y_pos)
    y_neg = sigm(y_neg)
    model = Model(inputs=[user, texts_pos, texts_neg], outputs=[y_pos, y_neg])

    loss = custom_loss(y_pos, y_neg, model_params)

    # Add loss to model
    model.add_loss(loss)
    return model


def idea1_architecture(embedding_size, model_params):
    initializer = tf.keras.initializers.RandomNormal(seed=1)
    texts_pos = Input(shape=(embedding_size,))
    texts_neg = Input(shape=(embedding_size,))

    sigmoid = Dense(1, activation='sigmoid', bias_initializer=initializers.Zeros(),
                    kernel_initializer=initializer)

    n_out = texts_neg
    p_out = texts_pos
    drop = Dropout(model_params['dropout_first'])
    n_out = drop(n_out)
    p_out = drop(p_out)

    for layer_size in model_params['layers']:
        drop = Dropout(model_params['dropout'])
        if model_params['reg'] != 0:
            layer = Dense(layer_size, activation=tf.nn.relu, bias_initializer=initializers.Zeros(),
                          kernel_initializer=initializer,
                          kernel_regularizer=regularizers.l2(model_params['reg']))
        else:
            layer = Dense(layer_size, bias_initializer=initializers.Zeros(), kernel_initializer=initializer,
                          activation=tf.nn.relu)
        n_out = drop(layer(n_out))
        p_out = drop(layer(p_out))

    y_pos = sigmoid(p_out)
    y_neg = sigmoid(n_out)
    model = Model(inputs=[texts_pos, texts_neg], outputs=[y_pos, y_neg])

    loss = custom_loss(y_pos, y_neg, model_params)

    # Add loss to model
    model.add_loss(loss)
    return model


# todo make nicer
class IntervalEvaluation(Callback):
    def __init__(self, test, seq, user_embedding, article_embedding, user_item_train, model_params, seq_validation,
                 model_path="tmp"):
        super(Callback, self).__init__()
        self.user_item_train = user_item_train
        self.seq_validation = seq_validation
        self.seq = seq
        self.test = test
        self.model_path = model_path
        self.user_embedding = user_embedding
        self.article_embedding = article_embedding
        self.model_params = model_params

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

        self.ndcg100 = -np.Inf
        self.recall10 = -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None, N=300):
        results = self.model.evaluate(self.seq_validation, batch_size=100)
        if not os.path.exists(f'results/{self.model_path}'):
            os.makedirs(f'results/{self.model_path}')
        with open(f"results/{self.model_path}/log", mode='+a') as w:
            w.writelines(str(results) + "\n")

        if not os.path.exists(f'results/{self.model_path}'):
            os.makedirs(f'results/{self.model_path}')
        with open(f"results/{self.model_path}/log", mode='+a') as w:
            w.writelines(str((logs.get("loss"), logs.get("val_loss"))) + "\n")

        if (epoch % self.model_params['interval'] == 0 and self.model_params['round'] == 0) or (
                self.model_params['round'] and self.model_params['round'] % self.model_params['interval'] == 0):

            pred, pred_raw = prediction(self.model, self.user_embedding, self.article_embedding, self.user_item_train,
                                        N, model_params=self.model_params, gen=self.seq, filter=True)
            pred = pred[pred.index.isin(self.test.index)]
            idea1 = evaluate(pred.sort_index(), self.test.loc[pred.index].sort_index(),
                             experiment_name=f'{self.model_path}/metrics', limit=1000)

            # early stopping
            # todo restore best weights if run till the end
            ndcg100 = idea1[4]
            recall10 = idea1[3]
            if self.model_params['early_stopping'] != False and self.model_params['stop_on_metric']:
                if ndcg100 - self.ndcg100 > 0.0005 or recall10 - self.recall10 > 0.0005:
                    if ndcg100 - self.ndcg100 > 0.0005:
                        self.ndcg100 = ndcg100

                    if recall10 - self.recall10 > 0.0005:
                        self.recall10 = recall10
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.model_params['early_stopping']:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        self.model.set_weights(self.best_weights)

        if epoch % self.model_params['checkpoint_interval'] == 0 and (self.model_params['round'] == 0):
            if not os.path.exists(f'{self.model_path}_epochs'):
                os.makedirs(f'{self.model_path}_epochs')
            self.model.save(f'{self.model_path}_epochs/{epoch}.h5')
        if self.model_params['round'] % self.model_params['checkpoint_interval'] == 0 and self.model_params['round']:
            if not os.path.exists(f'{self.model_path}_epochs'):
                os.makedirs(f'{self.model_path}_epochs')
            self.model.save(f'{self.model_path}_epochs/{self.model_params["round"]}.h5')


class data_generator(tf.keras.utils.Sequence):
    """
    Class to create and feed the batches. For each user-item sample in the training set sample 1
    negative sample i.e. articles he did not read. Returns batches of size batch_size where each sample is:
    ((uservector, pos_sample_article_vector),(uservector, neg_sample_article_vector))
    """

    def __init__(self, grouped_exploded, uservector, lookup, model_params, neg_samples=None):
        self.model_params = model_params
        self.lookup = lookup
        self.uservector = uservector
        if model_params['random_sampling']:
            self.grouped_upsampling = self.explode_data_and_add_original(grouped_exploded).sample(frac=1,
                                                                                                  random_state=1)
        else:
            self.grouped_upsampling = grouped_exploded.sample(frac=1, random_state=1)
            self.grouped_positives = self.grouped_upsampling.groupby('user_ix').resource_id.apply(set)
            self.grouped_upsampling = self.grouped_upsampling[
                self.grouped_upsampling['resource_id'].isin(self.lookup.index)]
            self.grouped_upsampling = self.grouped_upsampling[
                self.grouped_upsampling['user_ix'].isin(self.uservector.index)]

        self.neg_sampling = neg_samples
        self.new_batch_size = int(self.model_params['batch_size'] * 0.8)
        self.len = math.ceil(len(self.grouped_upsampling) / self.model_params['batch_size'])
        self.seen = 0

        self.uservector.columns = self.lookup.columns
        self.random_state = 0

    def __len__(self):
        return self.len

    def explode_data_and_add_original(self, grouped):
        """
        Helper function for upsamping
        """
        grouped_upsampling = pd.DataFrame(grouped.explode())
        grouped_upsampling.columns = ['pos_sample']
        grouped_upsampling['pos_samples'] = grouped
        grouped_upsampling = grouped_upsampling.reset_index()
        grouped_upsampling = grouped_upsampling[grouped_upsampling['pos_sample'].isin(self.lookup.index)]
        grouped_upsampling = grouped_upsampling[grouped_upsampling['user_ix'].isin(self.uservector.index)]
        return grouped_upsampling

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.random_state = self.random_state + 1
        self.grouped_upsampling = self.grouped_upsampling.sample(frac=1, random_state=self.random_state)

        gc.collect()

    def __getitem__(self, idx):

        k = idx % math.ceil(len(self.grouped_upsampling) / self.model_params['batch_size'])
        batch = self.grouped_upsampling.iloc[
                k * self.model_params['batch_size']:(k + 1) * self.model_params['batch_size'], :]

        # pick negative samples
        if self.model_params['random_sampling']:
            batch = self.random_sampling(batch,k)
        else:
            batch = self.timewise_sampling(batch)

        # create batch from user/articleids and normalize
        batch_final = self.create_batch(batch)

        for vector in batch_final:
            assert not np.any(np.isnan(vector))

        if self.model_params['epsilon']:
            for i in range(len(batch_final)):
                batch_final[i] = self.diff_privacy_noise_first(batch_final[i], self.model_params['epsilon'],
                                                               1.0 / len(self.grouped_upsampling))
        return (batch_final, batch_final[0])

    def create_batch(self, batch):

        users = batch['user_ix'].values
        pos_samples = batch['pos_sample'].values
        neg_samples = list(batch['neg_sample'])

        user = self.uservector.loc[users, :]
        pos = self.lookup.loc[pos_samples, :]
        neg = self.lookup.loc[neg_samples, :]

        user.index = batch.index
        pos.index = batch.index
        neg.index = batch.index

        if self.model_params['take_target_out']:
            user = (user.multiply(batch['pos_samples'].str.len(), axis='index').subtract(pos, axis='index')).divide(
                batch['pos_samples'].str.len() - 1, axis='index')

            if np.any(np.isnan(user)):
                user = self.uservector.loc[users, :]
                user.index = batch.index

        if self.model_params['normalize'] == 1:
            pos = np.concatenate((user.values, pos.values), axis=1)
            neg = np.concatenate((user.values, neg.values), axis=1)

            return (pos / np.linalg.norm(pos, ord=2, axis=1, keepdims=True),
                    neg / np.linalg.norm(neg, ord=2, axis=1, keepdims=True))

        if self.model_params['normalize'] == 2:
            user = user.values / np.linalg.norm(user.values, ord=2, axis=1, keepdims=True)
            pos = pos.values / np.linalg.norm(pos.values, ord=2, axis=1, keepdims=True)
            neg = neg.values / np.linalg.norm(neg.values, ord=2, axis=1, keepdims=True)
        else:
            user = user.values
            pos = pos.values
            neg = neg.values
        if type(self.model_params['layers'][0]) == list:  # idea2
            return (user, pos, neg)

        return (np.concatenate((user, pos), axis=1), np.concatenate((user, neg), axis=1))

    def timewise_sampling(self, batch):

        batch['neg_samples'] = [self.neg_sampling.get(str(key)) for key in batch['k']]
        batch['neg_sample'] = batch['neg_samples'].str[self.random_state % len(batch.iloc[0, :]['neg_samples'])]

        batch['neg_sample'] = batch['neg_sample'].astype(type(self.lookup.index[0]))

        batch['pos_sample'] = batch['resource_id']
        return batch

    def random_sampling(self, batch,k):

        # random sampling
        for i in range(9):
            batch['neg_sample'] = list(pd.Series(self.lookup.index).sample(n=len(batch), replace=True,
                                                                           random_state=self.random_state + k + i))
            batch_new = batch[[x[1] not in x[0] for x in zip(batch['pos_samples'], batch['neg_sample'])]]

            if len(batch_new) > self.new_batch_size or len(batch) == len(batch_new):
                break
        if i == 8:
            batch_new = batch_new.sample(n=self.new_batch_size, replace=True, random_state=self.random_state)
            print("Warning: had to use backup plan since we didnt find good negative samples")
        else:
            batch_new = batch_new.sample(n=min(len(batch_new), self.new_batch_size),
                                         random_state=self.random_state)
        return batch_new

    @staticmethod
    def diff_privacy_noise_first(X, epsilon, delta):
        diameter = (X.max() - X.min()).max()
        b = diameter / (epsilon - np.log(1 - delta))
        noise_vector = np.random.laplace(0, b, X.shape)
        return X + noise_vector


class test_data_generator(tf.keras.utils.Sequence):
    """
    For each user feed each article into the network
    """

    def __init__(self, uservector, lookup, model_params):
        self.lookup = lookup
        self.uservector = uservector
        self.normalize = model_params.get('normalize', 0)
        self.model_params = model_params

    def __len__(self):
        return len(self.uservector)

    def __getitem__(self, idx):
        user = self.uservector.iloc[idx, :]
        users = np.tile(user, (len(self.lookup), 1))
        articles = self.lookup.values
        if self.normalize == 1:
            sample = np.concatenate((users, articles), axis=1)
            sample = sample / np.linalg.norm(sample, ord=2, axis=1, keepdims=True)
            return ((sample, sample), sample)

        if self.normalize == 2:
            users = users / np.linalg.norm(users, ord=2, axis=1, keepdims=True)
            articles = articles / np.linalg.norm(articles, ord=2, axis=1, keepdims=True)

        if type(self.model_params['layers'][0]) == list:
            sample = (users, articles)
            return ((sample[0], sample[1], sample[1]),
                    sample)
        sample = np.concatenate((users, articles), axis=1)

        return ((sample, sample), sample)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass


def prediction(model, users, lookup, user_item_train, N, model_params=None, gen=None, filter=True):
    """
    Creates prediction for each article for each user in the testset. Filters out already read articles. Sorts the
    predicted score and returns the top N articles.
    """

    model_params['random_sampling'] = model_params.get('random_sampling', True)
    if gen is None:
        gen = test_data_generator(users, lookup, model_params)
    pred_raw = model.predict(gen, use_multiprocessing=False, verbose=1, workers=model_params['workers'])
    pred = pd.DataFrame(pred_raw[0])
    pred['article'] = lookup.index.fillna(-1).to_list() * len(users)
    pred.index = np.repeat(users.index, repeats=len(lookup))
    pred['read_articles'] = user_item_train
    pred_raw = pred.copy()
    isnull = pred['read_articles'].isnull()
    if isnull.sum() > 0:
        pred.loc[isnull, 'read_articles'] = [[[]] * isnull.sum()]
    if filter:
        pred = pred[[x[1] not in x[0] for x in zip(pred['read_articles'], pred['article'])]]
    pred = pred.reset_index()
    pred = pred.sort_values(0, ascending=False).groupby('user_ix').head(N).reset_index().sort_values(0, ascending=False)
    pred = pred.groupby('user_ix').apply(lambda x: list(x['article']))
    pred = pd.DataFrame(pred, columns=['predictions'])
    return pred, pred_raw


if __name__ == "__main__":
    from preprocessing import load_data, get_metadata, load_data_vertical

    N = 50  # number of predictions
    limit = 5000  # number of samples to look at
    first_time = True
    folder = os.getenv('DATA_FOLDER','processed')

    user_item_train, user_item_test, user_item_validation = load_data(folder=folder, cut=40)
    metadata = get_metadata(folder=folder, usecols=['resource_id', 'text'])  # slow
    user_item_train = user_item_train.head(limit)
    user_item_test = user_item_test.head(limit)

    print(f"Data loaded")
    model_params = {'lr': 0.0001,
                    'batch_size': 400,
                    'epochs': 4,
                    'alpha': 1,
                    'layers': [128, 32, 8],
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
                              model_path='idea1_models/random_sampling',
                              last_x_articles=1000)
    pred, raw = prediction(model, user_embedding.loc[user_item_train.head(limit).index],
                           article_embedding, user_item_train.head(limit), N, model_params=model_params)
    pred = pred[pred.index.isin(user_item_test.index)]
    idea1 = evaluate(pred.sort_index(), user_item_test.loc[pred.index].sort_index(),
                     experiment_name='idea1_random_sampling.results', limit=limit)
