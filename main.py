import argparse
from gmm import GaussianMixtureModel
from mdn import MixtureDensityNetwork
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

def gmm_hyperopt(train, test):
    components = [4, 9, 16, 25, 36]
    min_pitches = [0, 25, 100, 200]
    hypers = list(itertools.product(components, min_pitches))
    random.shuffle(hypers)
    for hyper in hypers:
        model = GaussianMixtureModel(components=hyper[0], min_pitches = hyper[1])
        model.fit(train)
        loglike = model.ptype_log_likelihood(test)
        print(hyper[0], hyper[1], loglike)

def mdn_hyperopt(train, test):
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    batch_sizes = [100, 500, 1000, 5000]
    player_embeddings = [10,20,30,40]
    iterationss = [1000, 5000]
    hidden_layerss = [[50], [75], [100], [100,50], [100,75], [100,75,50]]
    dropouts = [0.25, 0.5, 0.75]
    hypers = list(itertools.product(learning_rates, batch_sizes, player_embeddings, iterationss, hidden_layerss, dropouts))
    random.shuffle(hypers)
    print('Running Experiments')
    for hyper in hypers:
        learning_rate = hyper[0]
        batch_size = hyper[1]
        player_embedding = hyper[2]
        iterations = hyper[3]
        hidden_layers = hyper[4]
        dropout = hyper[5]
        model = MixtureDensityNetwork(learning_rate = learning_rate, 
                                        batch_size = batch_size,
                                        iterations = iterations,
                                        player_embedding = player_embedding,
                                        mixture_components = 9,
                                        hidden_layers = hidden_layers,
                                        dropout = dropout)
        model.fit(train)
        loglike = model.ptype_log_likelihood(test)
        print(learning_rate, batch_size, iterations, player_embedding, hidden_layers, dropout, loglike)                     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', default='/home/ryan/Desktop/') # raw data path
    parser.add_argument('--data', default='data.csv') # already preprocessed data
    parser.add_argument('--model', choices=['gmm', 'mdn'], default='gmm')
    parser.add_argument('--sample', default = None, type=int)
    parser.add_argument('--seed', type = int, default = None)
    parser.add_argument('--predict', choices=['ptype', 'ploc', 'both'], default='both')
    parser.add_argument('--hyperparam_opt', action='store_true')

    args = parser.parse_args() 

    pitches = pd.read_csv(args.data) if args.data else load_data(args.basepath)
   
    if args.sample: 
        pitches = pitches.sample(args.sample, random_state=args.seed)

    if args.hyperparam_opt:
        train, test = train_test_split(pitches)
        if args.model == 'gmm': 
            gmm_hyperopt(train, test)
        if args.model == 'mdn':
            mdn_hyperopt(train, test)
        sys.exit()
 
    if args.model == 'gmm':
        model = GaussianMixtureModel()
    elif args.model == 'mdn':
        model = MixtureDensityNetwork(dropout=0.5)

    model.fit(pitches)
    if args.predict == 'ptype':
        loglike = model.ptype_log_likelihood(pitches)
    elif args.predict == 'ploc':
        loglike = 'not implemented yet'
    else:
        loglike = model.log_likelihood(pitches)

    print(loglike)
       
 
