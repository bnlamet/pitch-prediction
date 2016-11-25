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

def load_data(base):
    games = pd.read_csv(base + 'games.csv').sort_values(by='date')
    atbats = pd.read_csv(base + 'atbats.csv') 
    pitches = pd.read_csv(base + 'pitches.csv')
    
    ptypes = ['FF','SL','FT','SI','CH','CU','FC','KC','FS','KN']
    pitches = pitches[pitches.type.map(lambda t: t in ptypes)]
    pitches = pitches[pitches.balls < 4]

    out = ['Strikeout', 'Groundout', 'Flyout', 'Pop Out', 'Lineout', 'Forceout', 'Grounded Into DP', 'Field Error', 'Sac Bunt', 'Sac Fly', 'Bunt Groundout', 'Fielders Choice Out', 'Bunt Pop Out', 'Strikout - DP', 'Fielders Choice', 'Sac Fly DP', 'Bunt Lineout', 'Triple Play', 'Sacrifice Bunt DP', 'Walk', 'Hit By Pitch', 'Intent Walk']
    hit = ['Single', 'Double', 'Triple', 'Home Run']
    # bad = ['Runner Out', 'Fan interference', 'Batter Interference', 'Catcher Interference']

    ball = ['Ball', 'Ball In Dirt', 'Intent Ball', 'Pitchout', 'Automatic Ball']
    strike = ['Called Strike', 'Swinging Strike', 'Foul Tip', 'Swinging Strike (Blocked)', 'Foul Bunt', 'Missed Bunt', 'Swinging Pitchout', 'Automatic Strike']
    foul = ['Foul', 'Foul (Runner Going)', 'Foul Pitchout']
    inplay = ['In play, out(s)', 'In play, no out', 'In play, run(s)', 'Hit By Pitch']

    des = lambda x: 'ball' if x in ball else 'strike' if x in strike else 'foul' if x in foul else 'inplay' if x in inplay else 'error'
    evt = lambda x: 'out' if x in out else 'hit' if x in hit else 'error'
    
    atbats.event = atbats.event.map(evt)
    atbats = atbats[atbats.event != 'error']

    pitches.des = pitches.des.map(des)
    pitches = pitches[pitches.des != 'error']
    pitches = pitches[(pitches.x > 0) & (pitches.y > 0)] # is this necessary?
    
    games['night'] = games.time_et.map(lambda s: dt.datetime.strptime(s, '%I:%M %p').time() > dt.time(18,0)).astype(int)
    # games['month'] = games.date.map(lambda s: dt.datetime.strptime(s, '%Y-%m-%d').month)

    atbats['home'] = atbats.inning_half.map({'top':0,'bottom':1})
    games = games[games.game_type == 'R']
    games['date'] = pd.to_datetime(games.date)
    pitches = pitches[pd.notnull(pitches.type)]
    atbats = games.merge(atbats, on='game_id')
    
    atbats['weekday'] = atbats.date.map(lambda t: t.weekday())
    atbats['month'] = atbats.date.map(lambda t: t.month)
    
    atbats.loc[atbats.home==0, 'batter_team'] = atbats[atbats.home==0].away_team
    atbats.loc[atbats.home==1, 'batter_team'] = atbats[atbats.home==1].home_team
    atbats.loc[atbats.home==0, 'pitcher_team'] = atbats[atbats.home==0].home_team
    atbats.loc[atbats.home==1, 'pitcher_team'] = atbats[atbats.home==1].away_team

    pitches = atbats.merge(pitches, on=['game_id', 'ab_num'])
    pitches.loc[pitches.des=='inplay', 'des'] = pitches.loc[pitches.des=='inplay', 'event']

    return pitches

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
       
 
