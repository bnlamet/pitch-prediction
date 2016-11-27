import argparse
from model1 import SimpleCategorical, GaussianMixtureModel
from model2 import CategoricalNeuralNetwork
#from gmm import GaussianMixtureModel
#from mdn import MixtureDensityNetwork
import datetime as dt
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import itertools
import random

def model1_hyperopt(train, test):
    components = [4, 9, 16, 25, 36]
    alphas = [1.0, 2.0, 4.0, 8.0, 16.0, 100000.0]
    hypers = list(itertools.product(alphas))
    random.shuffle(hypers)
    for hyper in hypers:
        model = SimpleCategorical(alpha = hyper[0])
        model.fit(train)
        loglike = model.log_likelihood(test)
        print(hyper, loglike)

def model2_hyperopt(train, test):
    learning_rates = [0.1, 0.5]
    batch_sizes = [1000, 5000, 10000]
    player_embeddings = [10, 20, 30, 40]
    sweepss = [50, 100, 250, 500]
    hidden_layerss = [[50], [75], [100], [100,50], [100,75], [100,75,50]]
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
    hypers = list(itertools.product(learning_rates, batch_sizes, player_embeddings, sweeps, hidden_layerss, dropouts))
    random.shuffle(hypers)
    print('Running Experiments')
    for hyper in hypers:
        model = CategoricalNeuralNetwork(learning_rate = hyper[0], 
                                            batch_size = hyper[1],
                                            sweeps = hyper[3],
                                            player_embedding = hyper[2],
                                            hidden_layers = hyper[4],
                                            dropout = hyper[5])
        model.fit(train)
        loglike = model.log_likelihood(test)
        print(hyper, loglike)                     

def model2_randsearch(train, test, tests=100):
    for _ in range(tests):
        learning_rate = 0
        batch_size = 0
        player_embedding = 0
        sweeps = 100
        hidden_layers = [100,75,50]
        dropout = 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', default='/home/ryan/Desktop/') # raw data path
    parser.add_argument('--data', default='data.csv') # already preprocessed data
    parser.add_argument('--model', choices=['model1', 'model2'], default='model1')
    parser.add_argument('--sample', default = None, type=int)
    parser.add_argument('--seed', type = int, default = None)
    parser.add_argument('--predict', choices=['ptype', 'ploc', 'both'], default='ptype')
    parser.add_argument('--gridsearch', action='store_true')

    args = parser.parse_args() 

    pitches = pd.read_csv(args.data) if args.data else load_data(args.basepath)
   
    if args.sample: 
        pitches = pitches.sample(args.sample, random_state=args.seed)

    train, test = train_test_split(pitches, random_state=args.seed)

    if args.gridsearch:
        if args.model == 'model1': 
            model1_hyperopt(train, test)
        if args.model == 'model2':
            model2_hyperopt(train, test)
        sys.exit()

    if args.model == 'model1':
        if args.predict == 'ptype':
            model = SimpleCategorical()
        elif args.predict == 'ploc':
            model = GaussianMixtureModel()
    elif args.model == 'model2':
        if args.predict == 'ptype':
            model = CategoricalNeuralNetwork(learning_rate = 0.1, batch_size = 1000, sweeps=100, player_embedding = 30, hidden_layers = [100, 75, 50], dropout = 0.1)
        elif args.predict == 'ploc':
            print('not implemented yet')    
            sys.exit()

    model.fit(train)
    loglike = model.log_likelihood(test)
    print(loglike)
       
 
