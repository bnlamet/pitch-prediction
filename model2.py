import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import pdb
import sys
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
tf.logging.set_verbosity(tf.logging.INFO)

class CategoricalNeuralNetwork: 
    def __init__(self, learning_rate = 0.1, 
                        batch_size = 500, 
                        sweeps = 50,
                        player_embedding = 30,
                        hidden_layers = [75],
                        dropout = 0.0, 
                        show_progress = False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sweeps = sweeps
        self.player_embedding = player_embedding
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.show_progress = show_progress

    def __setup_network(self):
        self.network = {}
      
        keep_prob = tf.placeholder(tf.float32)
 
        p_embeddings = tf.Variable(tf.random_uniform([self.n_pitchers, self.player_embedding], -1.0, 1.0))
        b_embeddings = tf.Variable(tf.random_uniform([self.n_batters, self.player_embedding], -1.0, 1.0))

        cat_batch = tf.placeholder(tf.int32, [None, self.n_cat - 1])
        real_batch = tf.placeholder(tf.float32, [None, self.n_real])
        type_batch = tf.placeholder(tf.int32, [None])
        alpha = tf.placeholder(tf.float32) # learning rate

        n_items = tf.shape(cat_batch)[0]

        p_embed = tf.nn.embedding_lookup(p_embeddings, cat_batch[:,0])
        b_embed = tf.nn.embedding_lookup(b_embeddings, cat_batch[:,1])
       
        # is tf.one_hot behavior consistent with different inputs? 
        inputs = [p_embed, b_embed, real_batch]
        for i in range(2, self.n_cat-1):
            inputs.append(tf.one_hot(cat_batch[:,i], depth=self.depths[i],on_value=1.0, off_value=0.0, dtype=tf.float32))
        input_layer = tf.concat(concat_dim=1, values=inputs)

        in_dim = input_layer.get_shape()[1]

        def create_variable(shape):
            # heuristic from page 295 of Deep Learning (Goodfellow, Bengio, and Courville)
            B = tf.sqrt(6.0 / tf.cast(tf.reduce_sum(shape), tf.float32))
            return tf.Variable(tf.random_uniform(shape, -B, B))

        # + 0.5 to handle the dead units problem with relu activation
        b = [create_variable([self.hidden_layers[0]]) + 0.5]
        W = [tf.Variable(tf.zeros([in_dim, self.hidden_layers[0]]))] # why does only zeros work here?
        hidden = [tf.nn.dropout(tf.nn.relu(tf.matmul(input_layer, W[0]) + b[0]), keep_prob)]
        for i in range(1, len(self.hidden_layers)):
            W.append(create_variable([self.hidden_layers[i-1], self.hidden_layers[i]]))
            b.append(create_variable([self.hidden_layers[i]]) + 0.5)
            hidden.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden[-1], W[-1]) + b[-1]), keep_prob))
       
        W.append(create_variable([self.hidden_layers[-1], self.n_types]))
#        b.append(create_variable([self.n_types]))
        # heuristic #1 from page 297 of Deep Learning
        b.append(tf.Variable(tf.log(tf.cast(1.0+self.marginals, tf.float32))))
        
        y_pred = tf.nn.softmax(tf.matmul(hidden[-1], W[-1]) + b[-1])
        y_true = tf.one_hot(indices=type_batch, depth=self.n_types, on_value=1.0, off_value=0.0, dtype=tf.float32)

#        This is in theory the correct way to calculate the cross entropy       
#        indices = tf.transpose(tf.pack([tf.range(0, n_items), type_batch]))
#        likelihoods = tf.gather_nd(y_pred, indices) 
#        loglike = tf.reduce_mean(tf.log(likelihoods))
 
        loglike = tf.reduce_mean(tf.reduce_sum(y_true * tf.log(y_pred + 1.0e-10), reduction_indices=[1])) 
#        train_step = tf.train.GradientDescentOptimizer(alpha).minimize(-loglike)
        # Adam is robust to hyperparameter settings (I think)
        train_step = tf.train.AdamOptimizer().minimize(-loglike)

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        self.network['keep_prob'] = keep_prob
        self.network['cat_batch'] = cat_batch
        self.network['real_batch'] = real_batch
        self.network['type_batch'] = type_batch
        self.network['y_pred'] = y_pred
        self.network['loglike'] = loglike
        self.network['train_step'] = train_step
        self.network['sess'] = sess

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
        self.marginals = np.bincount(self.cat_data[:, -1])

    def fit(self, pitches):
        self.__init_data(pitches) 
        self.__setup_network()
        cat_batch = self.network['cat_batch']
        real_batch = self.network['real_batch']
        type_batch = self.network['type_batch']
        train_step = self.network['train_step']
        keep_prob = self.network['keep_prob']
        sess = self.network['sess']
        alpha = self.network['alpha']
        loglike = self.network['loglike']

        for i in range(self.sweeps):
            perm = np.random.permutation(self.pitches.shape[0])
            cat_data = self.cat_data[perm]
            real_data = self.real_data[perm]
            # decaying learning rate as suggested by page 287 of Deep Learning
            if i < 100:
                alpha_val = (1.0 + i/100.0 + 0.01*i/100.0) * self.learning_rate
            else:
                alpha_val = self.learning_rate * 0.01
            for start in range(0, len(perm) - self.batch_size, self.batch_size):
                end = start + self.batch_size
                input_data = {  cat_batch : cat_data[start:end,:-1], 
                                real_batch : real_data[start:end],  
                                type_batch : cat_data[start:end,-1],
                                keep_prob : 1 - self.dropout,
                                alpha : alpha_val }
                sess.run(train_step, feed_dict = input_data)
            if self.show_progress:
                score = sess.run(loglike, feed_dict = { cat_batch : cat_data[:,:-1], 
                                                        real_batch : real_data,  
                                                        type_batch : cat_data[:,-1],
                                                        keep_prob : 1.0 })
                print(score)

    def log_likelihood(self, pitches):
        cat_data = np.array(list(self.prep.transform(pitches[self.cat_features])))
        real_data = pitches[self.real_features].values

        cat_batch = self.network['cat_batch']
        real_batch = self.network['real_batch']
        type_batch = self.network['type_batch']
        train_step = self.network['train_step']
        keep_prob = self.network['keep_prob']
        loglike = self.network['loglike']
        sess = self.network['sess']
 
        input_data = {  cat_batch : cat_data[:, :-1],
                        real_batch : real_data,
                        type_batch : cat_data[:,-1],
                        keep_prob : 1.0 }

        if False:
            y_pred = self.network['y_pred']
            preds = sess.run(y_pred, feed_dict = input_data)
            pdb.set_trace()

        return sess.run(loglike, feed_dict = input_data)


class MixtureDensityNetwork: 
    def __init__(self, learning_rate = 0.1, 
                        batch_size = 500, 
                        sweeps = 50,
                        player_embedding = 30,
                        hidden_layers = [75],
                        dropout = 0.0, 
                        mixture_components=16,
                        show_progress = False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sweeps = sweeps
        self.player_embedding = player_embedding
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.mixture_components = mixture_components
        self.show_progress = show_progress

    def __setup_network(self):
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

        W1 = tf.Variable(tf.zeros([in_dim, 75]))
        b1 = tf.Variable(tf.random_normal([75], 0.0, .01)) + 0.03

        hidden = tf.nn.relu(tf.matmul(input_layer, W1) + b1)

        n_mixtures = self.mixture_components
        W2 = tf.Variable(tf.random_normal([75, n_mixtures], 0.0, .01))
        b2 = tf.Variable(tf.random_normal([n_mixtures], 0.0, .01))
        W3 = tf.Variable(tf.random_normal([75, n_mixtures*2], 0.0, .01))
        b3 = tf.Variable(tf.random_normal([n_mixtures*2], 0.0, .01))
        W4 = tf.Variable(tf.random_normal([75, n_mixtures*2], 0.0, .01))
        b4 = tf.Variable(tf.random_normal([n_mixtures*2], 0.0, .01))

        weights = tf.nn.softmax(tf.matmul(hidden, W2) + b2)
        means =  tf.reshape(tf.matmul(hidden, W3) + b3, tf.pack([n_items, n_mixtures, 2]))
        stdevs = tf.reshape(tf.exp(tf.matmul(hidden, W4) + b4), tf.pack([n_items, n_mixtures, 2]))

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

    def fit(self, pitches):
        self.__init_data(pitches) 
        self.__setup_network()
        cat_batch = self.network['cat_batch']
        real_batch = self.network['real_batch']
        loc_batch = self.network['loc_batch']
        train_step = self.network['train_step']
        sess = self.network['sess']
        loglike = self.network['loglike']

        for i in range(self.sweeps):
            perm = np.random.permutation(self.pitches.shape[0])
            cat_data = self.cat_data[perm]
            real_data = self.real_data[perm]
            loc_data = self.loc_data[perm]
            for start in range(0, len(perm) - self.batch_size, self.batch_size):
                end = start + self.batch_size
                input_data = {  cat_batch : cat_data[start:end], 
                                real_batch : real_data[start:end],  
                                loc_batch : loc_data[start:end] }
                sess.run(train_step, feed_dict = input_data)
 
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
 
        input_data = {  cat_batch : cat_data,
                        real_batch : real_data,
                        type_batch : loc_data }

        return sess.run(loglike, feed_dict = input_data)


