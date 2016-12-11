import numpy as np
import pandas as pd
from transform_data import Transformer
from sklearn.linear_model import LogisticRegression
import pdb

class Logistic:

    def __init__(self):
        self.sparse = ['pitcher_id', 'batter_id', 'away_team', 'home_team', 'year', 'b_stand',
                        'p_throws', 'inning_half', 'batter_team', 'pitcher_team', 
                        'inning', 'order', 'weekday', 'month', 'balls', 'strikes']
        self.dense = ['night', 'home', 'sz_top', 'sz_bot']

    def fit(self, pitches):
        self.transformer = Transformer(pitches, self.sparse, self.dense, min_points=10)
        sparse_data = self.transformer.pandas_to_scipy(pitches)
        self.model = LogisticRegression(solver='sag', multi_class='multinomial', n_jobs=-1, max_iter=100000)
        self.model.fit(sparse_data, pitches.type) 

    def log_likelihood(self, pitches):
        sparse_data = self.transformer.pandas_to_scipy(pitches)
        probas = self.model.predict_proba(sparse_data)
        idx = pd.Series(index=self.model.classes_, data=range(probas.shape[1]))
        indices = idx[pitches.type].values
        likelihoods = probas[np.arange(probas.shape[0]), indices]
        return np.mean(np.log(likelihoods))
        


