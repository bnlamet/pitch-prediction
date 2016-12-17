import argparse
import sys
import itertools
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from logistic import Logistic

from model1 import SimpleCategorical, GaussianMixtureModel
from model2 import CategoricalNeuralNetwork, MixtureDensityNetwork


def random_hidden_layers():
    uniforms = [np.random.uniform(0.0, 3.0)]
    while uniforms[-1] > 0.67 and len(uniforms) < 6:
        u = uniforms[-1]
        uniforms.append(min(3.0, np.random.uniform(0.0, 1.33*u)))
    return [int(20 * np.exp(u)) for u in uniforms]

def model2_find_structure(train, test):
    for _ in range(100):
        hidden_layers = random_hidden_layers()
        print(hidden_layers)
        model = CategoricalNeuralNetwork(learning_rate=1.0, batch_size=2000, sweeps=25,
                                         player_embedding=40, hidden_layers=hidden_layers,
                                         dropout=0.1)
        model.fit(train)
        loglike = model.log_likelihood(test)
        print(loglike)

def model1_hyperopt(train, test):
    components = [4, 9, 16, 25, 36, 49]
    alphas = [1.0, 2.0, 4.0, 8.0, 16.0, 100000.0]
    hypers = list(itertools.product(alphas, components))
    random.shuffle(hypers)
    for hyper in hypers:
        model = GaussianMixtureModel(alpha=hyper[0], components=hyper[1])
        model.fit(train)
        loglike = model.log_likelihood(test)
        print(hyper, loglike)

def model2_hyperopt(train, test):
    learning_rates = [1.0] # doesn't matter, using Adam
    batch_sizes = [500, 1000, 2000, 5000]
    player_embeddings = [20, 30, 40]
    sweepss = [100]
    hidden_layerss = [[100, 75, 50]] #[[50], [75], [100], [100,50], [100,75], [100,75,50]]
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
    hypers = list(itertools.product(learning_rates, batch_sizes, player_embeddings, sweepss,
                                    hidden_layerss, dropouts))
    random.shuffle(hypers)
    print('Running Experiments')
    for hyper in hypers:
        model = CategoricalNeuralNetwork(learning_rate=hyper[0],
                                         batch_size=hyper[1],
                                         sweeps=hyper[3],
                                         player_embedding=hyper[2],
                                         hidden_layers=hyper[4],
                                         dropout=hyper[5])
        model.fit(train)
        loglike = model.log_likelihood(test)
        print(hyper, loglike)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data', default='data.csv') # already preprocessed data
    PARSER.add_argument('--model', choices=['model1', 'model2', 'logistic'], default='model1')
    PARSER.add_argument('--sample', default=None, type=int)
    PARSER.add_argument('--seed', type=int, default=None)
    PARSER.add_argument('--predict', choices=['ptype', 'ploc'], default='ptype')
    PARSER.add_argument('--gridsearch', action='store_true')
    PARSER.add_argument('--batch_size', type=int, default=2000)
    PARSER.add_argument('--dropout', type=float, default=0.1)
    PARSER.add_argument('--mixture_components', type=int, default=10)

    ARGS = PARSER.parse_args()

    PITCHES = pd.read_csv(ARGS.data)

    if ARGS.sample:
        PITCHES = PITCHES.sample(ARGS.sample, random_state=ARGS.seed)

    TRAIN, TEST = train_test_split(PITCHES, random_state=ARGS.seed)

    if ARGS.gridsearch:
        if ARGS.model == 'model1':
            model1_hyperopt(TRAIN, TEST)
        if ARGS.model == 'model2':
            model2_find_structure(TRAIN, TEST)
#            model2_hyperopt(train, test)
        sys.exit()

    if ARGS.model == 'model1':
        if ARGS.predict == 'ptype':
            MODEL = SimpleCategorical()
        elif ARGS.predict == 'ploc':
            MODEL = GaussianMixtureModel() # -2.62724153634
    elif ARGS.model == 'model2':
        if ARGS.predict == 'ptype':
            MODEL = CategoricalNeuralNetwork(show_progress=True)
        elif ARGS.predict == 'ploc':
            MODEL = MixtureDensityNetwork(mixture_components=ARGS.mixture_components,
                                          hidden_layers=[256, 128, 64], batch_size=ARGS.batch_size,
                                          player_embedding=50, dropout=ARGS.dropout,
                                          show_progress=True)
    elif ARGS.model == 'logistic':
        assert ARGS.predict == 'ptype'
        MODEL = Logistic()

    MODEL.fit(TRAIN)
    print('Training Log Likelihood: ', MODEL.log_likelihood(TRAIN))
    print('Testing Log Likelihood: ', MODEL.log_likelihood(TEST))
    if ARGS.predict == 'ptype':
        MODEL.calibration_curve(TEST)
