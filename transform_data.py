import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
import pdb

# pd.options.__mode.__chained_assignment = None

class Transformer:

    def __init__(self, data, sparse_cols, dense_cols = None, min_points = 50):
        self.data = data
        self.sparse_cols = sparse_cols
        self.dense_cols = dense_cols
        self.min_points = min_points
        bct = data.batter_id.map(self.data.batter_id.value_counts())
        pct = data.pitcher_id.map(self.data.pitcher_id.value_counts())
        data.loc[pct < min_points, 'pitcher_id'] = 0
        data.loc[bct <  min_points, 'batter_id'] = 0
        self.options = {}
        idx = 0
        self.maps = {}
        for col in sparse_cols:
            self.options[col] = self.data[col].unique()
            self.maps[col] = {}
            for option in self.options[col]:
                self.maps[col][option] = idx
                idx += 1
        self.encoder = None

    def pandas_to_numpy(self, pandas):
        copy = pandas.copy()
        for col in self.sparse_cols:
            copy.loc[:,col] = copy[col].map(self.maps[col])
        copy['batter_id'].fillna(self.maps['batter_id'][0], inplace=True)
        copy['pitcher_id'].fillna(self.maps['pitcher_id'][0], inplace=True)
        return copy[self.sparse_cols].values.astype(int)

    def _init_onehot(self):
        self.encoder = OneHotEncoder()
        numpy = self.pandas_to_numpy(self.data)
        self.encoder.fit(numpy)

    def pandas_to_scipy(self, pandas):
        if self.encoder is None:
            self._init_onehot()
        numpy = self.pandas_to_numpy(pandas)
        sparse = self.encoder.transform(numpy)
        if self.dense_cols:
            extra = pandas[self.dense_cols].values
            sparse = scipy.sparse.hstack([sparse, extra], format='csr')
        return sparse

if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    sparse = ['pitcher_id', 'batter_id', 'away_team', 'home_team', 'year', 'b_stand',
              'p_throws', 'inning_half', 'batter_team', 'pitcher_team', 'type']
    dense = ['night', 'inning', 'order', 'home', 'weekday',
             'month', 'balls', 'strikes', 'sz_top', 'sz_bot']
    t = Transformer(data, sparse, dense, min_points = 10)
    np = t.pandas_to_numpy(data)
    sp = t.pandas_to_scipy(data)
