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
import os

import pandas as pd

if __name__ == "__main__":

    import pandas as pd
    import numpy as np

    num_articles=99 ### must be <99
    num_users=10
    num_user_item_entries=1000
    folder = os.getenv('DATA_FOLDER','processed')
    if not os.path.exists(folder):
        os.makedirs(folder)
    user_item = pd.DataFrame([np.random.randint(0, num_users, size=(num_user_item_entries)), np.random.randint(0, num_articles, size=(num_user_item_entries))]).T
    user_item=user_item.reset_index()
    user_item.columns = ['ts','user_ix', 'article_id']

    text=pd.read_csv('blindtext', sep=';').iloc[:num_articles,:].reset_index()
    text.columns=['resource_id','text']


    # Loads the data and saves it in
    text.to_csv(f'{folder}/meta.csv')
    user_item.to_parquet(f"{folder}/user_item_matrix_vertical.pq")
