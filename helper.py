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
