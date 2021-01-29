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

import pandas as pd


def restrict_articles_to_timeframe(user_item,meta_data, start_date=datetime.date(2019, 12, 1),horizontal=True):
    """
    (Used for Partner B)
    Restricts the articles and the user-item-matrix to articles that are published after start_date
    """

    meta_data['publication_last_updated'] = pd.to_datetime(meta_data['publication_last_updated'], utc=True).dt.date
    meta_data = meta_data[(meta_data['publication_last_updated'] > start_date) & (
            meta_data['publication_last_updated'] < datetime.date(2020, 3, 1))]
    if horizontal:
        user_item = user_item.apply(
            lambda x: [article for article in x if article in meta_data['resource_id'].values])
        user_item=user_item[user_item.str.len()>0]
    else:
        user_item=user_item[user_item['resource_id'].isin(meta_data['resource_id'].values)]
    return user_item,meta_data
