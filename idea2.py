from preprocessing import load_data, get_metadata
from evaluation import evaluate
from helper import restrict_articles_to_timeframe
import pandas as pd
import gc
import pickle
import numpy as np
import os
import datetime
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Dense, Input, Embedding, concatenate, Flatten, Dropout,Dot,Concatenate
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import math
import tensorflow as tf

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
    Creates an embedding of the articles and users and saves it
    """
    grouped = model_params.get('grouped', True)
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
    if not grouped:
        grouped_train = grouped_train.groupby('user_ix').resource_id.apply(lambda x: list(x))
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


def nn_train(train, test, user_embedding, article_embedding, model_params, model_path, new=False, last_x_articles=1000,
             retrain=False):
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
    grouped = model_params.get('grouped', True)
    now = datetime.datetime.now()
    model = None
    if not new:
        model = tf.keras.models.load_model(f'{model_path}')
        history = ""

    if new or retrain:
        if grouped:

            val = train.str[-last_x_articles:].head(int(len(train) * 0.1))
            train = train.str[-last_x_articles:].tail(len(train) - int(len(train) * 0.1))
        else:
            val = train.head(int(len(train) * 0.1))
            train = train.tail(len(train) - int(len(train) * 0.1))
        model, history = nn_model(article_embedding, user_embedding, train, val, test, model_params,
                                  model_path=model_path, model=model)
        # model=None

    return model, history


import random


class data_generator(tf.keras.utils.Sequence):
    """
    Class to create and feed the batches. For each user-item sample in the training set samples 1 negative samples per user
    i.e. articles he did not read. Returns batches of size batch_size where each sample is:
    ((uservector, pos_sample_article_vector),(uservector, neg_sample_article_vector))
    """

    def __init__(self, grouped_exploded, uservector, lookup, model_params):
        self.sampling = model_params['random_sampling']
        self.grouped = model_params['grouped']
        self.take_target_out = model_params['take_target_out']
        self.recency = model_params['recency']
        self.lookup = lookup
        self.uservector = uservector
        if model_params['grouped']:
            self.grouped_upsampling = self.explode_data_and_add_original(grouped_exploded).sample(frac=1,
                                                                                                  random_state=1)
        else:
            self.grouped_upsampling = grouped_exploded.sample(frac=1, random_state=1)
            self.grouped_positives = self.grouped_upsampling.groupby('user_ix').resource_id.apply(lambda x: set(x))
            self.samples = self.grouped_upsampling.groupby('resource_id').last()
            #why was that here not needed for new sampling
            # self.grouped_positives.name = 'pos_samples'
            # self.grouped_upsampling = pd.merge(self.grouped_upsampling, self.grouped_positives, how='left',
            #                                    left_on='user_ix', right_index=True)
            self.grouped_upsampling = self.grouped_upsampling[
                self.grouped_upsampling['resource_id'].isin(self.lookup.index)]
            self.grouped_upsampling = self.grouped_upsampling[
                self.grouped_upsampling['user_ix'].isin(self.uservector.index)]

        self.batch_size = model_params['batch_size']
        self.normalize = model_params['normalize']
        self.new_batch_size = int(self.batch_size * 0.8)
        print(len(self.grouped_upsampling))
        self.len = math.ceil(len(self.grouped_upsampling) / self.batch_size)
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
        # if self.seen >= self.len:
        self.grouped_upsampling = self.grouped_upsampling.sample(frac=1, random_state=self.random_state)
        #     self.seen = 0
        #     print('shuffeled')

        gc.collect()

    def __getitem__(self, idx):

        k = idx % math.ceil(len(self.grouped_upsampling) / self.batch_size)
        batch = self.grouped_upsampling.iloc[k * self.batch_size:(k + 1) * self.batch_size, :]

        if self.sampling:
            # try:
            if self.grouped:

                for i in range(9):
                    batch['neg_sample'] = list(pd.Series(self.lookup.index).sample(n=len(batch), replace=True,
                                                                                   random_state=self.random_state + k + i))
                    batch_new = batch[[x[1] not in x[0] for x in zip(batch['pos_samples'], batch['neg_sample'])]]

                    if len(batch_new) > self.new_batch_size or len(batch) == len(batch_new):
                        break
                if i == 8:
                    batch_new = batch_new.sample(n=self.new_batch_size, replace=True, random_state=self.random_state)
                    print("hmmmm")
                else:
                    batch_new = batch_new.sample(n=min(len(batch_new), self.new_batch_size),
                                                 random_state=self.random_state)
                batch = self.create_batch(batch_new)
                # except KeyError:
                #     print("hopla")
                #     return self.__getitem__(idx + 1)
            else:
                1
                #
                # for i in range(9):
                #     samples = self.samples.sample(n=len(batch), replace=True,
                #                                   random_state=self.random_state + k + i)
                #     batch['neg_sample'] = list(samples['article_id'])
                #     batch['neg_date'] = list(samples['publication_last_updated'])
                #     batch_new = batch[[x[1] not in x[0] for x in zip(batch['pos_samples'], batch['neg_sample'])]]
                #
                #     if len(batch_new) > self.new_batch_size or len(batch) == len(batch_new):
                #         break
                # if i == 8:
                #     batch_new = batch_new.sample(n=self.new_batch_size, replace=True, random_state=self.random_state)
                #     print("hmmmm")
                # else:
                #     batch_new = batch_new.sample(n=min(len(batch_new), self.new_batch_size),
                #                                  random_state=self.random_state)
                #
                # batch_new['pos_sample'] = batch['article_id']
                # batch_new['neg_published_at'] = pd.to_datetime(batch_new['publication_last_updated'])
                # batch_new['neg_recency'] = batch_new['published_at'] - batch_new['time']
                # batch_new['neg_recency'] = batch_new['neg_recency'].dt.days
                # # nzz_user_item_test2_merged.loc[nzz_user_item_test2_merged[nzz_user_item_test2_merged['recency']>0].index,'recency']=0
                # # todo here what to do
                # batch_pos, batch_neg = self.create_batch(batch_new)
        else:

            # batch['neg_samples']=[neg_samples for i in range(len(batch))]
            # batch['neg_sample']=neg_samples[0]

            # todo not fixed
            batch['neg_sample'] = batch['neg_samples'].str[self.random_state % 30]
            # #
            # #     batch['neg_sample']=[random.choice([neg for neg in neg_samples if neg not in pos]) for pos in batch['pos_samples']]
            # except IndexError:
            #     print("no neg samples found")
            #     batch['neg_sample'] = [random.choice([neg for neg in self.lookup.index if neg not in pos]) for pos in
            #                            batch['pos_samples']]

            batch['pos_sample'] = batch['resource_id']
            # batch = batch[['user_ix', 'pos_sample', 'neg_sample']]
            # print(batch)
            batch_pos, batch_neg = self.create_batch(batch)
        # if np.any(np.isnan(batch_pos)) or np.any(np.isnan(batch_neg)):
        #     pd.DataFrame(batch_pos).to_csv('test_p.csv')
        #     pd.DataFrame(batch_neg).to_csv('test_n.csv')
        #     pd.DataFrame(batch).to_csv('test2.csv')
        #     print("aa")
        assert not np.any(np.isnan(batch[0]))
        assert not np.any(np.isnan(batch[1]))
        assert not np.any(np.isnan(batch[2]))

        return (batch, batch[0])

    def create_batch(self, batch):

        users = batch['user_ix'].values
        pos_samples = batch['pos_sample'].values
        neg_samples = list(batch['neg_sample'])

        ## take pos sample out
        user = self.uservector.loc[users, :]
        pos = self.lookup.loc[pos_samples, :]
        neg = self.lookup.loc[neg_samples, :]

        user.index = batch.index
        pos.index = batch.index
        neg.index = batch.index

        # if self.take_target_out:
        #     user = (user.multiply(batch['pos_samples'].str.len(), axis='index').subtract(pos, axis='index')).divide(
        #         batch['pos_samples'].str.len() - 1, axis='index')
        #
        #     if np.any(np.isnan(user)):
        #         user = self.uservector.loc[users, :]
        #         user.index = batch.index
        #         print("test")
        #
        # if self.normalize == 1:
        #     pos = np.concatenate((user.values, pos.values),
        #                          axis=1)
        #     neg = np.concatenate((user.values, neg.values),
        #                          axis=1)
        #
        #     return pos / np.linalg.norm(pos, ord=2, axis=1, keepdims=True), neg / np.linalg.norm(neg, ord=2, axis=1,
        #                                                                                          keepdims=True)
        #
        # if self.normalize == 2:
        #     return np.concatenate(
        #         (user.values / np.linalg.norm(user.values, ord=2, axis=1, keepdims=True),
        #          pos.values / np.linalg.norm(pos.values, ord=2, axis=1, keepdims=True)),
        #         axis=1), np.concatenate(
        #         (user.values / np.linalg.norm(user.values, ord=2, axis=1, keepdims=True),
        #          neg.values / np.linalg.norm(neg.values, ord=2, axis=1, keepdims=True)), axis=1)
        # # return np.multiply(self.uservector.loc[users, :].values, self.lookup.loc[pos_samples, :].values), \
        # #        np.multiply(self.uservector.loc[users, :].values, self.lookup.loc[neg_samples, :].values),

        return (user.values, pos.values,                neg.values)
        # return np.concatenate((user.values, pos.values),
        #                       axis=1), \
        #        np.concatenate((user.values, neg.values),
        #                       axis=1)


from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras import initializers


def nn_model(lookup, uservector, train, val, test, model_params, model_path, model):
    """
    Defines the model and trains it
    """
    if len(lookup) != 2:
        lookup = (lookup, lookup)
    model_params['grouped'] = model_params.get('grouped', True)
    model_params['random_sampling'] = model_params.get('random_sampling', 1)
    model_params['lr'] = model_params.get('lr', 0.00001)
    model_params['batch_size'] = model_params.get('batch_size', 100)
    model_params['epochs'] = model_params.get('epochs', 2)
    model_params['alpha'] = model_params.get('alpha', 1)
    model_params['layers'] = model_params.get('layers', [1024, 512, 8])
    model_params['dropout'] = model_params.get('dropout', 0.5)
    model_params['dropout_first'] = model_params.get('dropout_first', model_params['dropout'])
    model_params['reg'] = model_params.get('reg', 0)
    model_params['normalize'] = model_params.get('normalize', 0)
    model_params['interval'] = model_params.get('interval', 50000)
    model_params['eval_batch'] = model_params.get('eval_batch', True)
    model_params['checkpoint_interval'] = model_params.get('checkpoint_interval', 50000)
    model_params['early_stopping'] = model_params.get('early_stopping', 0)
    model_params['loss'] = model_params.get('loss', "0")
    model_params['optimizer'] = model_params.get('optimizer', "ADAM")
    model_params['take_target_out'] = model_params.get('take_target_out', True)
    model_params['decay'] = model_params.get('decay', False)
    model_params['recency'] = model_params.get('recency', False)
    model_params['stop_on_metric']=model_params.get('stop_on_metric', False)
    if model_params['grouped']:
        train_grouped=train
        model_params['steps'] = model_params.get('steps',
                                                 int(train.str.len().sum() / model_params['batch_size']) )

        test_ids = test
    else:
        #todo only last 10 for test
        train_grouped=pd.concat([train[train['user_ix'].isin(test['user_ix'])],val[val['user_ix'].isin(test['user_ix'])]]).groupby(
            'user_ix').resource_id.apply(lambda x: list(x))
        test_ids = test.groupby('user_ix').resource_id.apply(lambda x: list(x)).head(1000)
        model_params['steps'] = model_params.get('steps', int(len(train) / model_params['batch_size']))

    if model:
        model.save(f'{model_path}_backup')
    else:
        def custom_loss(y_pred_pos, y_pred_neg, model_params):

            alpha = K.constant(model_params['alpha'])
            pointwise_loss = -K.log(y_pred_pos + 1e-07) - K.log(1 - y_pred_neg + 1e-07)

            if model_params['loss'] == 'TOP':
                pairwise_loss = K.sigmoid(y_pred_neg - y_pred_pos) + K.sigmoid(y_pred_neg * y_pred_neg)

            elif model_params['loss'] == 'BPR_r':
                pairwise_loss = -K.log(
                    K.sigmoid(y_pred_pos - y_pred_neg) + 1 - 0.73105857863)  # 0.26894142137->0.73105857863
            # max loss=-0.26931230852
            else:
                pairwise_loss = -K.log(
                    K.sigmoid(y_pred_pos - y_pred_neg) + 1e-07)  # +1-0.73105857863) #0.26894142137->0.73105857863
            loss = alpha * pairwise_loss + (1 - alpha) * pointwise_loss

            return tf.reduce_mean(loss)

        embedding_size = lookup[0].shape[1]


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

        # dotted = Dot(1, normalize=True)
        #
        # y_pos = dotted([u_out, p_out])
        # y_neg = dotted([u_out, n_out])

        concat=Concatenate()
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

        y_pos=sigm(y_pos)
        y_neg=sigm(y_neg)
        model = Model(inputs=[user, texts_pos, texts_neg], outputs=[y_pos, y_neg])

        loss = custom_loss(y_pos, y_neg, model_params)

        # Add loss to model
        model.add_loss(loss)
        if model_params['optimizer'] == "ADAM":
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=model_params['lr']))  # ,clipvalue=0.5 ))
        else:
            model.compile(optimizer=tf.keras.optimizers.SGD(lr=model_params['lr']))  # ,clipvalue=0.5) )
        model.summary()
    print(model_params)
    with tf.device('/gpu:0'):
        seq_train = data_generator(train, lookup=lookup[0], uservector=uservector, model_params=model_params)
        tmp = model_params['take_target_out']
        model_params['take_target_out'] = False
        seq_validation = data_generator(test, lookup=lookup[0], uservector=uservector, model_params=model_params)
        seq_test = test_data_generator(uservector.loc[test_ids.index], lookup[1], model_params=model_params)
        model_params['take_target_out'] = tmp

        ival = IntervalEvaluation(test=test_ids, seq=seq_test, user_embedding=uservector.loc[test_ids.index],
                                  article_embedding=lookup[1],
                                  user_item_train=train_grouped,
                                  model_params=model_params,
                                  model_path=model_path, seq_validation=seq_validation)
        if model_params['early_stopping'] and not model_params['stop_on_metric']:
            patience = model_params['early_stopping']
        else:
            patience = 1000

        def scheduler(epoch, lr):
            base_lr = model_params['lr']
            if epoch % model_params['decay'][0] == 0:
                print("ergewrgwergwreg")
                lr = lr * model_params['decay'][1]
            return max(0.0000001, lr)

        callback = [ival,
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, mode='min',
                                                     patience=patience,
                                                     restore_best_weights=True)]

        if model_params['decay']:
            callback.append(LearningRateScheduler(scheduler))
        history = model.fit(seq_train, validation_data=seq_validation, callbacks=callback,
                            epochs=model_params['epochs'], steps_per_epoch=model_params['steps'], verbose=1,
                            max_queue_size=100, shuffle=False,
                            use_multiprocessing=True, workers=1)

        model.save(f'{model_path}')
        del seq_train
        del seq_validation
        del seq_test
    return model, history


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
        self.eval_batch = model_params['eval_batch']
        self.interval = model_params['interval']
        self.checkpoint_interval = model_params['checkpoint_interval']

        self.stop_on_metric = model_params['stop_on_metric']
        self.patience = model_params['early_stopping']
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

    def on_batch_end(self, batch, logs=None, N=300):
        if self.eval_batch:
            if batch % self.interval == 0:
                print("\nlr:")
                print(self.model.optimizer.lr.numpy())
                try:
                    results = self.model.evaluate(self.seq_validation, batch_size=100)
                    try:
                        if not os.path.exists(f'results/{self.model_path}'):
                            os.makedirs(f'results/{self.model_path}')
                        with open(f"results/{self.model_path}/log", mode='+a') as w:
                            w.writelines(str(results) + "\n")
                        # pd.DataFrame(loss).to_csv(f"results/{self.model_path}/log", mode='+a')
                    except Exception as error:
                        print(error)
                    pred_raw = self.model.predict(self.seq, verbose=1)
                    pred = pd.DataFrame(pred_raw[0])
                    pred['article'] = self.article_embedding.index.fillna(-1).to_list() * len(self.user_embedding)
                    pred.index = np.repeat(self.user_embedding.index, repeats=len(self.article_embedding))
                    pred['read_articles'] = self.user_item_train
                    isnull = pred['read_articles'].isnull()
                    if isnull.sum() > 0:
                        pred.loc[isnull, 'read_articles'] = [[[]] * isnull.sum()]
                    pred = pred[[x[1] not in x[0] for x in zip(pred['read_articles'], pred['article'])]]
                    pred = pred.reset_index()
                    pred = pred.sort_values(0, ascending=False).groupby('user_ix').head(N).reset_index().sort_values(0, ascending=False)
                    pred = pred.groupby('user_ix').apply(lambda x: list(x['article']))
                    pred = pd.DataFrame(pred, columns=['predictions'])
                    pred = pred[pred.index.isin(self.test.index)]
                    idea1 = evaluate(pred.sort_index(), self.test.loc[pred.index].sort_index(),
                                     experiment_name=f'{self.model_path}/metrics', limit=1000)
                except Exception as error:
                    print(error)
            if batch % self.checkpoint_interval == 0:
                if not os.path.exists(f'{self.model_path}_epochs'):
                    os.makedirs(f'{self.model_path}_epochs')
                self.model.save(f'{self.model_path}_epochs/{batch}')

    def on_epoch_end(self, epoch, logs=None, N=300):
        results = self.model.evaluate(self.seq_validation, batch_size=100)
        try:
            if not os.path.exists(f'results/{self.model_path}'):
                os.makedirs(f'results/{self.model_path}')
            with open(f"results/{self.model_path}/log", mode='+a') as w:
                w.writelines(str(results) + "\n")
            # pd.DataFrame(loss).to_csv(f"results/{self.model_path}/log", mode='+a')
        except Exception as error:
            print(error)
        try:

            if not os.path.exists(f'results/{self.model_path}'):
                os.makedirs(f'results/{self.model_path}')
            with open(f"results/{self.model_path}/log", mode='+a') as w:
                w.writelines(str((logs.get("loss"), logs.get("val_loss"))) + "\n")
            # pd.DataFrame(loss).to_csv(f"results/{self.model_path}/log", mode='+a')
        except Exception as error:
            print(error)
        if not self.eval_batch:
            if epoch % self.interval == 0:

                pred_raw = self.model.predict(self.seq, verbose=1)
                pred = pd.DataFrame(pred_raw[0])
                pred['article'] = self.article_embedding.index.fillna(-1).to_list() * len(self.user_embedding)
                pred.index = np.repeat(self.user_embedding.index, repeats=len(self.article_embedding))
                pred['read_articles'] = self.user_item_train
                # self.test.to_pickle('test1')
                # isnull = pred['read_articles'].isnull()
                # if isnull.sum() > 0:
                #     pred.loc[isnull, 'read_articles'] = [[[]] * isnull.sum()]
                # pred = pred[[x[1] not in x[0] for x in zip(pred['read_articles'], pred['article'])]]
                pred = pred.reset_index()
                pred = pred.sort_values(0, ascending=False).groupby('user_ix').head(N).reset_index().sort_values(0, ascending=False)
                pred = pred.groupby('user_ix').apply(lambda x: list(x['article']))
                pred = pd.DataFrame(pred, columns=['predictions'])
                pred = pred[pred.index.isin(self.test.index)]
                idea1 = evaluate(pred, self.test.loc[pred.index],
                                 experiment_name=f'{self.model_path}/metrics', limit=1000)

                #early stopping
                ndcg100=idea1[4]
                recall10=idea1[3]
                if self.patience!=False and self.stop_on_metric:
                    if  ndcg100-self.ndcg100>0.0005 or recall10-self.recall10>0.0005:
                        if ndcg100-self.ndcg100>0.0005:
                            self.ndcg100=ndcg100

                        if recall10-self.recall10>0.0005:
                            self.recall10=recall10
                        self.wait = 0
                        self.best_weights = self.model.get_weights()
                    else:
                        self.wait += 1
                        if self.wait >= self.patience:
                            self.stopped_epoch = epoch
                            self.model.stop_training = True
                            self.model.set_weights(self.best_weights)
                    print(self.wait)
                    print(self.recall10)
                    print(self.ndcg100)
            if epoch % self.checkpoint_interval == 0:
                if not os.path.exists(f'{self.model_path}_epochs'):
                    os.makedirs(f'{self.model_path}_epochs')
                self.model.save(f'{self.model_path}_epochs/{epoch}')


class test_data_generator(tf.keras.utils.Sequence):

    def __init__(self, uservector, lookup, model_params):
        self.lookup = lookup
        self.uservector = uservector
        self.normalize = model_params.get('normalize', 0)

    def __len__(self):
        return len(self.uservector)

    def __getitem__(self, idx):
        user = self.uservector.iloc[idx, :]

        sample = (np.tile(user, (len(self.lookup), 1)), self.lookup.values)
        #
        # sample = np.multiply(np.tile(user, (len(self.lookup), 1)), self.lookup.values)
        if self.normalize == 1:
            sample = (sample[0] / np.linalg.norm(sample[0], ord=2, axis=1, keepdims=True), sample[1] / np.linalg.norm(
                sample[1], ord=2, axis=1, keepdims=True),sample[1] / np.linalg.norm(
                sample[1], ord=2, axis=1, keepdims=True))
            return (sample,
                    sample)

        return ((sample[0],sample[1],sample[1]),
                sample)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass


def prediction(model, users, lookup, user_item_train, N, model_params=None, gen=None, filter=True):
    """
    Creates prediction for each article for each user in the testset. Filters out already read articles. Sorts the
    predicted score and returns the top N articles.
    """

    model_params['grouped'] = model_params.get('grouped', True)
    if gen is None:
        gen = test_data_generator(users, lookup, model_params)
    pred_raw = model.predict(gen, use_multiprocessing=False, verbose=1, workers=8)
    pred = pd.DataFrame(pred_raw[0])
    pred['article'] = lookup.index.fillna(-1).to_list() * len(users)
    pred.index = np.repeat(users.index, repeats=len(lookup))
    # if model_params['grouped']:
    pred['read_articles'] = user_item_train
    # else:
    #     pred['read_articles'] = user_item_train.groupby('user_ix').resource_id.apply(lambda x: list(x))
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
    from preprocessing import load_data, get_metadata
    from evaluation import evaluate
    from helper import restrict_articles_to_timeframe

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
