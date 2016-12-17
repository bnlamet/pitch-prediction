import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import pdb
import sys
from calibration_curve import plot_curve
    
# P(type | C)
class SimpleCategorical:
 
    def __init__(self, alpha = 1.0):
        self.alpha = alpha

    def fit(self, pitches):
        self.prior = pitches.type.value_counts(normalize=True)
        counts = pitches.pitcher_id.value_counts()
        self.table = pitches.groupby('pitcher_id') \
                            .apply(lambda sample: sample.type.value_counts()) \
                            .unstack() \
                            .fillna(0)[self.prior.index]
        self.table = self.table + self.alpha*self.prior
        for col in self.table.columns:
            self.table[col] /= (counts + self.alpha)

    def log_likelihood(self, pitches):
        loglike = 0.0
        for (pid, ptype), group in pitches.groupby(['pitcher_id', 'type']):
            if pid in self.table.index:
                p = self.table.loc[pid, ptype]
            else:
                p = self.prior[ptype]
            loglike += len(group) * np.log(p)
        return loglike / pitches.shape[0]

    def calibration_curve(self, pitches):
        pids = pitches.pitcher_id.values
        probas = self.table.loc[pids].fillna(self.prior)
        labels = list(probas.columns)
        actuals = pitches.type.map(labels.index).values
        plot_curve(labels, probas.values.transpose(), actuals, 50)


# P(x, y | type, C)
class GaussianMixtureModel:

    def __init__(self, components = 25, alpha = 1.0):
        self.components = components
        self.alpha = alpha
        self.models = {} # self.models[(pitcher_id, pitch_type)] = gaussian mixture model
        self.counts = {}

    def fit(self, pitches):
        self.priors = {}
        for ptype, group in pitches.groupby('type'):
            self.priors[ptype] = self.__fit_mixture(group[['px','pz']].values)
        for idx, group in pitches.groupby(['pitcher_id', 'type']):
            points = group[['px', 'pz']].values
            self.models[idx] = self.__fit_mixture(points)
            self.counts[idx] = group.shape[0]

    def __fit_mixture(self, points):
        model = GaussianMixture(n_components = min(points.shape[0], self.components))
        model.fit(points)
        return model

    def log_likelihood(self, pitches):
        loglike = 0.0
        for idx, group in pitches.groupby(['pitcher_id', 'type']):
            points = group[['px', 'pz']].values
            if idx in self.models:
                A = self.models[idx].score_samples(points)
                B = self.priors[idx[1]].score_samples(points)
                p = self.counts[idx] * np.exp(A) + self.alpha * np.exp(B)
                p /= self.counts[idx] + self.alpha
                loglike += np.log(p).sum()
            else:
                loglike += self.priors[idx[1]].score_samples(points).sum()
        return loglike / pitches.shape[0]
