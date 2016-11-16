import numpy as np
import pandas as pd
from scipy import stats

class GaussianMixtureModel:
 
    def __init__(self, components = 25, min_pitches = 0):
        self.components = components
        self.min_pitches = min_pitches
        self.models = {} # self.models[pitcher_id][pitch_type] = probablity, gaussian mixture model

    def fit(self, pitches):
        self.bounds = (pitches.px.min(), pitches.px.max(), pitches.pz.min(), pitches.pz.max())
        pcounts = pitches.pitcher_id.value_counts()
        self.pitchers = list(pcounts[pcounts > self.min_pitches].index) + [0]
        pitches['pitcher_id'] = pitches.pitcher_id.map(lambda p: p if p in self.pitchers else 0)
        for pid, group in pitches.groupby('pitcher_id'):
            probas = group.type.value_counts(normalize=True)
            self.models[pid] = {}
            for ptype, sample in group.groupby('type'):
                points = sample[['px', 'pz']].values
                self.models[pid][ptype] = probas[ptype], self.__fit_mixture(points)

    # stub
    def __fit_mixture(self, points):
        normal = stats.multivariate_normal(mean=[0,0], cov = [ [1,0], [0,1] ])
        return normal.pdf

    def log_likelihood(self, pitches):
        likelihoods = self.__likelihoods(pitches)
        return likelihoods.map(np.log).sum() 

    def __likelihoods(self, pitches):
        pitches['pitcher_id'] = pitches.pitcher_id.map(lambda p: p if p in self.pitchers else 0)
        return pitches.apply(self.__likelihood, axis=1)

    def __likelihood(self, pitch):
        pitcher = pitch.pitcher_id
        ptype = pitch.type
        x, y = pitch.px, pitch.pz
        p, gmm = self.models[pitcher][ptype]
        return p * gmm((x,y))

