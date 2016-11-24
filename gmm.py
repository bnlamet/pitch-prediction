import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import pdb
import sys

# P(type, x, y | C) = P(type, x, y | pitcher)
# P(type, x, y) = P(type) P(x, y | type)
# P(type) approximated using sample proportions
# P(x, y | type) is approximated by numerically learning parameters to a GMM using gradient based methods
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

    def __fit_mixture(self, points):
        model = GaussianMixture(n_components = min(points.shape[0], self.components))
        model.fit(points)
        return model
        
    def log_likelihood(self, pitches):
        # note: this mutates data frame
        pitches['pitcher_id'] = pitches.pitcher_id.map(lambda p: p if p in self.pitchers else 0)
        loglike = 0.0
        for pid, group in pitches.groupby('pitcher_id'):
            for ptype, sample in group.groupby('type'):
                if ptype in self.models[pid]:
                    p, gmm = self.models[pid][ptype]
                    loglike += len(sample)*(np.log(p) + gmm.score(sample[['px', 'pz']].values))
                else:
                    return np.inf
        return loglike / pitches.shape[0]

    def ptype_log_likelihood(self, pitches):
        pitches['pitcher_id'] = pitches.pitcher_id.map(lambda p: p if p in self.pitchers else 0)
        loglike = 0.0
        for pid, group in pitches.groupby('pitcher_id'):
            for ptype, sample in group.groupby('type'):
                if ptype in self.models[pid]:
                    p, gmm = self.models[pid][ptype]
                    loglike += len(sample) * np.log(p)
                else:
                    return np.inf
        return loglike / pitches.shape[0]


