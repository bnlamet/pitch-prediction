import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import pdb
import sys
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from sklearn.model_selection import train_test_split
# tf.logging.set_verbosity(tf.logging.INFO)

class CategoricalNeuralNetwork: 
    def __init__(self, learning_rate = 0.1, 
                        batch_size = 500, 
                        sweeps = 50,
                        player_embedding = 30,
                        hidden_layers = [75],
                        dropout = 0.0,
                        activation = 'relu', 
                        patience = 8,
                        show_progress = False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sweeps = sweeps
        self.player_embedding = player_embedding
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.patience = patience
        self.show_progress = show_progress
        if activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif activation == 'tanh':
            self.activation = tf.nn.tanh

    def __setup_network(self):
      
        keep_prob = tf.placeholder(tf.float32)
        p_embeddings = tf.Variable(tf.random_uniform([self.n_pitchers, self.player_embedding], -1.0, 1.0))
        b_embeddings = tf.Variable(tf.random_uniform([self.n_batters, self.player_embedding], -1.0, 1.0))

        pitcher_batch = tf.placeholder(tf.int32, [None])
        batter_batch = tf.placeholder(tf.int32, [None]) 
        cat_batch = tf.placeholder(tf.float32, [None, self.cat_width]) # one hot encoding
        real_batch = tf.placeholder(tf.float32, [None, self.n_real])
        type_batch = tf.placeholder(tf.float32, [None, self.n_types]) # one hot encoding
        alpha = tf.placeholder(tf.float32) # learning rate

        n_items = tf.shape(cat_batch)[0]

        p_embed = tf.nn.embedding_lookup(p_embeddings, pitcher_batch)
        b_embed = tf.nn.embedding_lookup(b_embeddings, batter_batch)
       
        input_layer = tf.concat(concat_dim=1, values=[p_embed, b_embed, real_batch, cat_batch])
#        for i in range(2, self.n_cat-1):
#            inputs.append(tf.one_hot(cat_batch[:,i], depth=self.depths[i],on_value=1.0, off_value=0.0, dtype=tf.float32))
#        input_layer = tf.concat(concat_dim=1, values=inputs)

        in_dim = input_layer.get_shape()[1]

        def create_variable(shape):
            # heuristic from page 295 of Deep Learning (Goodfellow, Bengio, and Courville)
            B = tf.sqrt(6.0 / tf.cast(tf.reduce_sum(shape), tf.float32))
            return tf.Variable(tf.random_uniform(shape, -B, B))

        # + 0.5 to handle the dead units problem with relu activation
        b = [create_variable([self.hidden_layers[0]]) + 0.5]
        W = [tf.Variable(tf.zeros([in_dim, self.hidden_layers[0]]))] # why does only zeros work here?
#        W = [create_variable([in_dim, self.hidden_layers[0]])]
        hidden = [tf.nn.dropout(self.activation(tf.matmul(input_layer, W[0]) + b[0]), keep_prob)]
        for i in range(1, len(self.hidden_layers)):
            W.append(create_variable([self.hidden_layers[i-1], self.hidden_layers[i]]))
            b.append(create_variable([self.hidden_layers[i]]) + 0.5)
            hidden.append(tf.nn.dropout(self.activation(tf.matmul(hidden[-1], W[-1]) + b[-1]), keep_prob))
       
        W.append(create_variable([self.hidden_layers[-1], self.n_types]))
        # heuristic #1 from page 297 of Deep Learning
        b.append(tf.Variable(tf.log(tf.cast(1.0+self.marginals, tf.float32))))
        
        type_pred = tf.nn.softmax(tf.matmul(hidden[-1], W[-1]) + b[-1])
#        y_true = tf.one_hot(indices=type_batch, depth=self.n_types, on_value=1.0, off_value=0.0, dtype=tf.float32)

#        This is in theory the correct way to calculate the cross entropy       
#        indices = tf.transpose(tf.pack([tf.range(0, n_items), type_batch]))
#        likelihoods = tf.gather_nd(y_pred, indices) 
#        loglike = tf.reduce_mean(tf.log(likelihoods))
 
        loglike = tf.reduce_mean(tf.reduce_sum(type_batch * tf.log(type_pred + 1.0e-10), reduction_indices=[1])) 
        # Adam is robust to hyperparameter settings (I think)
        train_step = tf.train.AdamOptimizer().minimize(-loglike)

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        self.network = {}
        self.network['keep_prob'] = keep_prob
        self.network['cat_batch'] = cat_batch
        self.network['real_batch'] = real_batch
        self.network['type_batch'] = type_batch
        self.network['batter_batch'] = batter_batch
        self.network['pitcher_batch'] = pitcher_batch
        self.network['type_pred'] = type_pred
        self.network['loglike'] = loglike
        self.network['train_step'] = train_step
        self.network['sess'] = sess
        self.network['init'] = init

    def one_hot(self, array, depth):
        ans = np.zeros((array.size, depth))
        ans[np.arange(array.size), array] = 1.0
        return ans

    def __init_data(self, pitches):
        self.cat_features = ['pitcher_id', 'batter_id', 'away_team', 'home_team', 'year', 'b_stand', 
                        'p_throws', 'inning_half', 'batter_team', 'pitcher_team', 
                        'order', 'weekday', 'month', 'balls', 'strikes', 'type']
        self.real_features = ['night', 'inning', 'order', 'home', 'weekday',
                        'month', 'balls', 'strikes', 'sz_top', 'sz_bot']
        self.pitches = pitches
        self.prep = learn.preprocessing.CategoricalProcessor()
        self.prep.fit(pitches[self.cat_features])
        self.cat_data = np.array(list(self.prep.transform(pitches[self.cat_features])))
        self.real_data = pitches[self.real_features].values
        self.n_cat = len(self.cat_features)
        self.n_real = len(self.real_features)
        self.depths = [len(pitches[col].unique())+1 for col in self.cat_features] 
        self.type_onehot = self.one_hot(self.cat_data[:,-1], self.depths[-1])
        self.cat_onehot = np.concatenate([self.one_hot(self.cat_data[:,i], self.depths[i]) for i in range(2, self.n_cat-1)], axis=1)
        self.pitcher_data = self.cat_data[:, 0]
        self.batter_data = self.cat_data[:, 1]
        self.n_pitchers = self.pitcher_data.max() + 1
        self.n_batters = self.batter_data.max() + 1
        self.n_types = self.cat_data[:,-1].max() + 1
        self.cat_width = self.cat_onehot.shape[1]
       # note 'year' could have larger depth depending on data
        # +1 accounts for possibility of unobserved category
        self.marginals = np.bincount(self.cat_data[:, -1]) 

    def training_sweep(self, perm):
        sess = self.network['sess']
        for start in range(0, perm.size - self.batch_size, self.batch_size):
            end = start + self.batch_size
            idx = perm[start:end]
            input_data = { self.network['cat_batch'] : self.cat_onehot[idx],
                            self.network['real_batch'] : self.real_data[idx],
                            self.network['type_batch'] : self.type_onehot[idx],
                            self.network['batter_batch'] : self.batter_data[idx],
                            self.network['pitcher_batch'] : self.pitcher_data[idx],
                            self.network['keep_prob'] : 1 - self.dropout }
            sess.run(self.network['train_step'], feed_dict = input_data)

    def fit(self, pitches):
        self.__init_data(pitches) 
        self.__setup_network()

        sess = self.network['sess'] 
        train, valid = train_test_split(np.arange(self.pitches.shape[0]))
        sweep = 0
        best_sweeps = 0
        best_score = -np.inf 

        valid_data = { self.network['cat_batch'] : self.cat_onehot[valid],
                        self.network['real_batch'] : self.real_data[valid],
                        self.network['type_batch'] : self.type_onehot[valid],
                        self.network['batter_batch'] : self.batter_data[valid],
                        self.network['pitcher_batch'] : self.pitcher_data[valid],
                        self.network['keep_prob'] : 1.0 }


        while sweep <= best_sweeps + self.patience:
            sweep += 1
            perm = np.random.permutation(train)
            self.training_sweep(perm)

            score = sess.run(self.network['loglike'], feed_dict = valid_data)
            if self.show_progress:
                print(score)
            if score > best_score:
                best_sweeps = sweep
                best_score = score

        if self.show_progress:
            print('Stopped Early... retraining on all data')

        sess = tf.Session()
        self.network['sess'] = sess
        sess.run(self.network['init'])

        for i in range(best_sweeps):
            perm = np.random.permutation(self.pitches.shape[0])
            self.training_sweep(perm)
            if self.show_progress:
                print('Iteration %d of %d' % (i, best_sweeps))


    def log_likelihood(self, pitches):
        cat_data = np.array(list(self.prep.transform(pitches[self.cat_features])))
        real_data = pitches[self.real_features].values
        pitcher_data = cat_data[:,0]
        batter_data = cat_data[:,1]
        type_onehot = self.one_hot(cat_data[:,-1], self.depths[-1])
        cat_onehot = np.concatenate([self.one_hot(cat_data[:,i], self.depths[i]) for i in range(2, self.n_cat-1)], axis=1)

        batter_batch = self.network['batter_batch']
        pitcher_batch = self.network['pitcher_batch']
        cat_batch = self.network['cat_batch']
        real_batch = self.network['real_batch']
        type_batch = self.network['type_batch']
        train_step = self.network['train_step']
        keep_prob = self.network['keep_prob']
        loglike = self.network['loglike']
        sess = self.network['sess']
 
        input_data = {  cat_batch : cat_onehot,
                        real_batch : real_data,
                        type_batch : type_onehot,
                        batter_batch : batter_data,
                        pitcher_batch : pitcher_data,
                        keep_prob : 1.0 }

        return sess.run(loglike, feed_dict = input_data)


class MixtureDensityNetwork: 
    def __init__(self, learning_rate = 0.1, 
                        batch_size = 500, 
                        sweeps = 50,
                        player_embedding = 30,
                        hidden_layers = [75],
                        dropout = 0.0, 
                        mixture_components=16,
                        patience = 5,
                        show_progress = False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sweeps = sweeps
        self.player_embedding = player_embedding
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.mixture_components = mixture_components
        self.patience = patience
        self.show_progress = show_progress

    def __setup_network(self):

        keep_prob = tf.placeholder(tf.float32)
        p_embeddings = tf.Variable(tf.random_uniform([self.n_pitchers, self.player_embedding], -1.0, 1.0))
        b_embeddings = tf.Variable(tf.random_uniform([self.n_batters, self.player_embedding], -1.0, 1.0))

        cat_batch = tf.placeholder(tf.int32, [None, self.n_cat])
        real_batch = tf.placeholder(tf.float32, [None, self.n_real])
        loc_batch = tf.placeholder(tf.float32, [None, 2])

        n_items = tf.shape(cat_batch)[0]

        pitchers = cat_batch[:,0]
        batters = cat_batch[:,1]

        p_embed = tf.nn.embedding_lookup(p_embeddings, pitchers)
        b_embed = tf.nn.embedding_lookup(b_embeddings, batters)
        inputs = [p_embed, b_embed, real_batch]
        for i in range(2, self.n_cat):
            inputs.append(tf.one_hot(cat_batch[:,i], depth=self.depths[i],
                        on_value=1.0, off_value=0.0, dtype=tf.float32))
        input_layer = tf.concat(concat_dim=1, values=inputs)

        in_dim = input_layer.get_shape()[1]

        def create_variable(shape):
            # heuristic from page 295 of Deep Learning (Goodfellow, Bengio, and Courville)
            B = tf.sqrt(6.0 / tf.cast(tf.reduce_sum(shape), tf.float32))
            return tf.Variable(tf.random_uniform(shape, -B, B))

        # + 0.5 to handle the dead units problem with relu activation
        b = [tf.Variable(tf.zeros([self.hidden_layers[0]])) + 0.5]
        W = [tf.Variable(tf.zeros([in_dim, self.hidden_layers[0]]))] # why does only zeros work here?
        hidden = [tf.nn.dropout(tf.nn.relu(tf.matmul(input_layer, W[0]) + b[0]), keep_prob)]
        for i in range(1, len(self.hidden_layers)):
            W.append(create_variable([self.hidden_layers[i-1], self.hidden_layers[i]]))
            b.append(tf.Variable(tf.zeros([self.hidden_layers[i]]) + 0.5))
            hidden.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden[-1], W[-1]) + b[-1]), keep_prob))
 
        n_mixtures = self.mixture_components
        n_hidden = self.hidden_layers[-1]
        W2 = create_variable([n_hidden, n_mixtures])
        b2 = tf.Variable(tf.zeros([n_mixtures]))
        W3 = create_variable([n_hidden, n_mixtures*2])
        b3 = tf.Variable(tf.zeros([n_mixtures*2]))
        W4 = create_variable([n_hidden, n_mixtures*2])
        b4 = tf.Variable(tf.zeros([n_mixtures*2]))

        weights = tf.nn.softmax(tf.matmul(hidden[-1], W2) + b2)
        means = tf.reshape(tf.matmul(hidden[-1], W3) + b3, tf.pack([n_items, n_mixtures, 2]))
        stdevs = tf.reshape(tf.exp(tf.matmul(hidden[-1], W4)+b4), tf.pack([n_items, n_mixtures, 2]))

        def likelihood(i):
            normals = tf.contrib.distributions.MultivariateNormalDiag(means[i], stdevs[i])
            return tf.reduce_sum(weights[i] * normals.pdf(loc_batch[i]))

        likelihoods = tf.map_fn(likelihood, tf.range(0, n_items), dtype=tf.float32)
        loglike = tf.reduce_mean(tf.log(likelihoods))

        train_step = tf.train.AdamOptimizer().minimize(-loglike)
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        self.network = {}
        self.network['cat_batch'] = cat_batch
        self.network['real_batch'] = real_batch
        self.network['loc_batch'] = loc_batch
        self.network['loglike'] = loglike
        self.network['train_step'] = train_step
        self.network['sess'] = sess
        self.network['keep_prob'] = keep_prob
        self.network['init'] = init

    def __init_data(self, pitches):
        self.cat_features = ['pitcher_id', 'batter_id', 'away_team', 'home_team', 'year', 'b_stand', 
                        'p_throws', 'inning_half', 'batter_team', 'pitcher_team', 
                        'order', 'weekday', 'month', 'balls', 'strikes', 'type']
        self.real_features = ['night', 'inning', 'order', 'home', 'weekday',
                        'month', 'balls', 'strikes', 'sz_top', 'sz_bot']
        self.pitches = pitches
        self.prep = learn.preprocessing.CategoricalProcessor()
        self.prep.fit(pitches[self.cat_features])
        self.cat_data = np.array(list(self.prep.transform(pitches[self.cat_features])))
        self.real_data = pitches[self.real_features].values
        self.loc_data = pitches[['px','pz']].values
        self.n_pitchers = self.cat_data[:,0].max() + 1
        self.n_batters = self.cat_data[:,1].max() + 1
        self.n_types = self.cat_data[:,-1].max() + 1
        self.n_cat = len(self.cat_features)
        self.n_real = len(self.real_features)
        # note 'year' could have larger depth depending on data
        # +1 accounts for possibility of unobserved category
        self.depths = [len(pitches[col].unique())+1 for col in self.cat_features] 

    def training_sweep(self, perm):
        sess = self.network['sess']
        for start in range(0, perm.size - self.batch_size, self.batch_size):
            end = start + self.batch_size
            idx = perm[start:end]
            input_data = {  self.network['cat_batch'] : self.cat_data[idx], 
                            self.network['real_batch'] : self.real_data[idx],  
                            self.network['loc_batch'] : self.loc_data[idx], 
                            self.network['keep_prob'] : 1.0 - self.dropout }
            sess.run(self.network['train_step'], feed_dict = input_data)

    def fit(self, pitches):
        self.__init_data(pitches) 
        self.__setup_network()

        sess = self.network['sess'] 
        train, valid = train_test_split(np.arange(self.pitches.shape[0]))
        sweep = 0
        best_sweeps = 0
        best_score = -np.inf 

        valid_data = { self.network['cat_batch'] : self.cat_data[valid],
                        self.network['real_batch'] : self.real_data[valid],
						self.network['loc_batch'] : self.loc_data[valid], 
                        self.network['keep_prob'] : 1.0 }

        while sweep <= best_sweeps + self.patience:
            sweep += 1
            perm = np.random.permutation(train)
            self.training_sweep(perm)

            score = sess.run(self.network['loglike'], feed_dict = valid_data)
            if self.show_progress:
                print(score)
            if score > best_score:
                best_sweeps = sweep
                best_score = score

        if self.show_progress:
            print('Stopped Early... retraining on all data')

        sess = tf.Session()
        self.network['sess'] = sess
        sess.run(self.network['init'])

        for i in range(best_sweeps):
            perm = np.random.permutation(self.pitches.shape[0])
            self.training_sweep(perm)
            if self.show_progress:
                print('Iteration %d of %d' % (i, best_sweeps))

    def log_likelihood(self, pitches):
        cat_data = np.array(list(self.prep.transform(pitches[self.cat_features])))
        real_data = pitches[self.real_features].values
        loc_data = pitches[['px','pz']].values

        cat_batch = self.network['cat_batch']
        real_batch = self.network['real_batch']
        loc_batch = self.network['loc_batch']
        train_step = self.network['train_step']
        loglike = self.network['loglike']
        sess = self.network['sess']
        keep_prob = self.network['keep_prob']
 
        input_data = {  cat_batch : cat_data,
                        real_batch : real_data,
                        loc_batch : loc_data,
                        keep_prob : 1.0 }

        return sess.run(loglike, feed_dict = input_data)


