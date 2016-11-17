import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import pdb
import sys

class MixtureDensityNetwork:
 
    def __init__(self, components = 25):
        self.components = components
        self.models = {} # self.models[pitcher_id][pitch_type] = probablity, gaussian mixture model

    def fit(self, pitches):
        return None

