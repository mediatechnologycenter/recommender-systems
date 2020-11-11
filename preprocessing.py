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

import pandas as pd

"""
This module is mainly used to transform the data from the partners into our desired format.
In the and only load_data and get_metadata is used in the algorithms.
"""


def load_data(folder, input_path='user_item', cut=40):
    """
    loads the training,validation,test set from the folder, restricts the users with at least "cut" read articles and
    returns the sets. The Format of the sets is pd.Series with index the UserID and value a list of ArticleIDs
    :param folder/input_path: {folder}/{input_path} is the path to look for the *_train.pkl files
    :param cut: value to cut off users with less than "cut" read articles
    :return: three pd.Series. Index of each series is the UserID. The value is a list of ArticleIDs.
    (look in create_split to see how the split is defines)
    """
    # cut cuts off users that read less than cut articles

    user_item_train, user_item_test, user_item_validation = pd.read_pickle(
        f'{folder}/{input_path}_train.pkl'), pd.read_pickle(f'{folder}/{input_path}_test.pkl'), pd.read_pickle(
        f'{folder}/{input_path}_validation.pkl')

    user_item_train = user_item_train[user_item_train.str.len() > cut * 0.7]
    user_item_test = user_item_test.loc[user_item_train.index]
    user_item_validation = user_item_validation.loc[user_item_train.index]

    return user_item_train, user_item_test, user_item_validation


def load_data_cv(folder, input_path='user_item', cut=40):
    """
    Same as load_data but only returns random 80% of the training set
    """
    # cut cuts off users that read less than cut articles
    user_item_train, user_item_test, user_item_validation = load_data(folder, input_path=input_path, cut=cut)
    user_item_train = user_item_train.sample(frac=0.8)
    return user_item_train, user_item_test, user_item_validation


def get_metadata(folder, usecols=None):
    """
    Loads and returns the article metadata.
    The algorithms expect the format to be a Dataframe with two columns:
    - "resource_id": unique id for the article
    - "text": full text of the article (without html tags)
    """
    if not usecols:
        usecols = ['text', 'resource_id']
    metadata = pd.read_csv(f"{folder}/meta.csv", usecols=usecols)

    # if  not usecols or 'lead' in usecols:
    #     metadata['text']=metadata['lead']
    # if 'lead_text' in usecols:
    #     metadata['text']=metadata['lead_text']
    return metadata.dropna(subset=['text'])


def save_metadata_partner_b(input_folder, output_folder):
    """
    Ignore: Specific to one dataset
    (Processes and saves the article metadata into a cleaned format)
    """
    json_files_in = sorted([os.path.join(input_folder, filename) for filename in os.listdir(input_folder)])
    metadata = []
    for i, json_file_in in enumerate(json_files_in):
        print("Processing " + json_file_in)
        df = pd.read_json(json_file_in, orient='records')
        df['text'] = df['parsed_content'].apply(lambda x: "\n".join([p['text'] for p in x]))
        metadata.append(df)
    metadata = pd.concat(metadata, sort=True)

    metadata['resource_id'] = metadata['article_id'].str[3:].astype(int)
    metadata.to_csv(f'{output_folder}/meta.csv')
    return metadata


def save_user_item_matrix_partner_a(input_folder, output_folder, file_name='user_item_matrix_vertical.pq'):
    """
    Ignore: Specific to one dataset
    (Creates and stores a DataFrame with three columns:
    "user_ix", "article_id" and "ts" (a timestamp). Each user_ix article_id pair indicates a click of the user on the
    article at a time ts.)
    """
    now = datetime.datetime.now()
    dir = f'{input_folder}/meta.csv'
    data_raw = pd.read_csv(dir).dropna(subset=['text'])
    dir = f'{input_folder}/matrix.pq'
    matrix = pd.read_parquet(dir)
    dir = f'{input_folder}/events.pq'
    events = pd.read_parquet(dir)
    events['article_id'] = events['event_id'].str.split('/').str[-1]
    matrices = pd.merge(matrix, events[['event_ix', 'article_id']], 'left', left_on='event_ix',
                        right_on='event_ix')
    matrices = matrices[matrices['article_id'].isin(data_raw['resource_id'])]

    matrices.to_parquet(f'{output_folder}/{file_name}')

    print(f"data loaded {datetime.datetime.now() - now}")


def save_user_item_matrix_partner_b(input_folder, output_folder, file_name='user_item_matrix_vertical.pq'):
    """
    Ignore: Specific to one dataset
        (Creates and stores a DataFrame with three columns:
    "user_ix", "article_id" and "ts" (a timestamp). Each user_ix article_id pair indicates a click of the user on the
    article at a time ts.)
    """
    now = datetime.datetime.now()
    dir = f'{input_folder}/published_articles_2020-10-07-2337.parquet'
    data_raw = pd.read_parquet(dir, columns=['article_id'])
    dir = f'{input_folder}/user_read_articles_february_final.parquet'
    matrix = pd.read_parquet(dir, columns=['article_id', 'time', 'user_id_hashed'])
    dir = f'{input_folder}/user_read_articles_january_final.parquet'
    matrix2 = pd.read_parquet(dir, columns=['article_id', 'time', 'user_id_hashed'])
    data_raw['article_id'] = data_raw['article_id'].str[3:].astype(int)
    matrix['article_id'] = matrix['article_id'].str[3:].astype(int)
    matrix2['article_id'] = matrix2['article_id'].str[3:].astype(int)
    matrices = pd.concat([matrix, matrix2])
    matrices = matrices[matrices['article_id'].isin(data_raw['article_id'])]
    matrices.columns = [x if x != 'user_id_hashed' else 'user_ix' for x in matrices.columns]

    matrices.to_parquet(f'{output_folder}/{file_name}')
    print(f"data loaded {datetime.datetime.now() - now}")


def transform_item_matrix_to_horizontal_format(folder, output_path='user_item_matrix.pkl',
                                               input_path='user_item_matrix_vertical.pq', sortby='ts'):
    """
    Transforms vertical User-Item matrix where ich row is one click into a horizontal User-item matrix where we have
    one row for each user and each row contains a (sorted) list of articles she/he clicked on.
    :param folder: Input folder
    :param output_path: Filename/path for outputfile
    :param input_path: Filename/path for inputfile. This pickled file contains a DataFrame with three columns:
                        "user_ix": the UserID and "article_id" the ArticleID and "<sortby>" which should be timestamp
                        to sort by. Each UserID ArticleID pair indicates a click of the user on the article at a time.
    :param sortby: Columnname of the timestamp column to sort by
    :return: returns a Series where the index is the UserID and values is the by timestamp
             sorted list of clicked ArticleIDs
    """
    now = datetime.datetime.now()
    matrices = pd.read_parquet(f"{folder}/{input_path}")
    grouped = matrices.sort_values(sortby).groupby(['user_ix']).article_id.apply(lambda x: list(x))

    grouped.to_pickle(f"{folder}/{output_path}")
    print(f"Data transformed {datetime.datetime.now() - now}")


def create_split(folder, input_path='user_item_matrix.pkl', ouput_path='user_item', cut_dump=10):
    """
    Loads the horizontal user item data from folder and creates a user-wise a 70% train, 20% validation, 10% test split.
    This means for each user the first 70% read articles are in the train the next 20% in validation and the last 10%
    read articles in the test set. We remove users with less than 10 clicked articles.
    This is the data that is loaded to train/test the models in the end.
    """
    now = datetime.datetime.now()
    user_item = pd.read_pickle(f"{folder}/{input_path}")

    user_item = user_item[user_item.str.len() > (cut_dump)]

    user_item_train = user_item.apply(lambda x: x[:int(len(x) * 0.7)])
    user_item_test = user_item.apply(lambda x: x[int(len(x) * 0.7):int(len(x) * 0.9)])
    user_item_validation = user_item.apply(lambda x: x[int(len(x) * 0.9):])

    user_item_train.name = 'article_id'
    user_item_test.name = 'article_id'
    user_item_validation.name = 'article_id'

    user_item_train.to_pickle(f'{folder}/{ouput_path}_train.pkl')
    user_item_test.to_pickle(f'{folder}/{ouput_path}_test.pkl')
    user_item_validation.to_pickle(f'{folder}/{ouput_path}_validation.pkl')

    print(f"Split created {datetime.datetime.now() - now}")


def transform_horizontal_to_vertical(df):
    """
    Transforms the horizontal format into vertical format
    :param df:
    :return:
    """
    return df.explode().reset_index()


if __name__ == "__main__":

    import pandas as pd

    import numpy as np


    #run on dummy data dummy data
    if 'PARTNER_A_OUTPUT_FOLDER' not in os.environ:
        num_articles=99 ### must be <99
        num_users=10
        num_user_item_entries=1000

        user_item = pd.DataFrame([np.random.randint(0, num_users, size=(num_user_item_entries)), np.random.randint(0, num_articles, size=(num_user_item_entries))]).T
        user_item=user_item.reset_index()
        user_item.columns = ['ts','user_ix', 'article_id']

        text=pd.read_csv('blindtext', sep=';').iloc[:num_articles,:].reset_index()
        text.columns=['resource_id','text']
        folder = 'processed'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Loads the data and saves it in
        text.to_csv(f'{folder}/meta.csv')
        user_item.to_parquet(f"{folder}/user_item_matrix_vertical.pq")
        # Transforms the user-item-matrix into a user-series. For each user we store the articles read as one sorted list.
        # Save the new format.
        # This format is more convenient for creating the split and for training some of the algorithms.
        transform_item_matrix_to_horizontal_format(folder=folder)
        # Create a train,test,validation split. 70%,10%,20% and save it
        create_split(folder=folder, cut_dump=10)
        # loads the saved train,validation,test split
        train, test, validation = load_data(folder=folder, cut=40)
        # # if you wish to transform into normal user-item-format
        train_vertical = transform_horizontal_to_vertical(train)

    else:
        folder = os.environ['PARTNER_A_OUTPUT_FOLDER']
        input_folder = os.environ['PARTNER_A_INPUT_FOLDER']
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Loads the data and saves it in
        save_user_item_matrix_partner_a(input_folder=input_folder, output_folder=folder)
        # Transforms the user-item-matrix into a user-series. For each user we store the articles read as one sorted list.
        # Save the new format.
        # This format is more convenient for creating the split and for training some of the algorithms.
        transform_item_matrix_to_horizontal_format(folder=folder)
        # Create a train,test,validation split. 70%,10%,20% and save it
        create_split(folder=folder, cut_dump=10)
        # loads the saved train,validation,test split
        train, test, validation = load_data(folder=folder, cut=40)
        # # if you wish to transform into normal user-item-format
        train_vertical = transform_horizontal_to_vertical(train)

        folder = os.environ['PARTNER_B_OUTPUT_FOLDER']
        input_folder = os.environ['PARTNER_B_INPUT_FOLDER']
        if not os.path.exists(folder):
            os.makedirs(folder)
        # # Loads the data and saves it in
        save_metadata_partner_b(f"{input_folder}/cleaned", folder)
        save_user_item_matrix_partner_b(input_folder=input_folder, output_folder=folder)
        # Transforms the user-item-matrix into a user-series. For each user we store the articles read as one sorted list.
        # Save the new format.
        # This format is more convenient for creating the split and for training some of the algorithms.
        transform_item_matrix_to_horizontal_format(folder=folder, sortby='time')
        # Create a train,test,validation split. 70%,10%,20% and save it
        create_split(folder=folder, cut_dump=10)
        ## loads the saved train,validation,test split
        train, test, validation = load_data(folder=folder, cut=40)
        ## if you wish to transform into normal rowwise user-item format
        # train_vertical=transform_horizontal_to_vertical(train)