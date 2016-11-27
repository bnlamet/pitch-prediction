import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import pdb
import sys
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
#import progressbar

class CategoricalNeuralNetwork:

    def __init__(self, learning_rate = 0.1, 
                        batch_size = 500, 
                        sweeps = 50,
                        player_embedding = 30,
                        hidden_layers = [75],
                        dropout = 0.0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sweeps = sweeps
        self.player_embedding = player_embedding
        self.hidden_layers = hidden_layers
        self.dropout = dropout

    def __setup_network(self):
        self.network = {}
      
        keep_prob = tf.placeholder(tf.float32)
 
        p_embeddings = tf.Variable(tf.random_uniform([self.n_pitchers, self.player_embedding], -1.0, 1.0))
        b_embeddings = tf.Variable(tf.random_uniform([self.n_batters, self.player_embedding], -1.0, 1.0))

        cat_batch = tf.placeholder(tf.int32, [None, self.n_cat - 1])
        real_batch = tf.placeholder(tf.float32, [None, self.n_real])
        type_batch = tf.placeholder(tf.int32, [None])

        p_embed = tf.nn.embedding_lookup(p_embeddings, cat_batch[:,0])
        b_embed = tf.nn.embedding_lookup(b_embeddings, cat_batch[:,1])
       
        # is tf.one_hot behavior consistent with different inputs? 
        inputs = [p_embed, b_embed, real_batch]
        for i in range(2, self.n_cat-1):
            inputs.append(tf.one_hot(cat_batch[:,i], depth=self.depths[i],on_value=1.0, off_value=0.0, dtype=tf.float32))
        input_layer = tf.concat(concat_dim=1, values=inputs)

        in_dim = input_layer.get_shape()[1]

        def create_variable(shape):
            return tf.Variable(tf.random_normal(shape, 0.0, 0.1))

        b = [create_variable([self.hidden_layers[0]]) + 0.5]
        W = [tf.Variable(tf.zeros([in_dim, self.hidden_layers[0]]))] # why does only zeros work here?
        hidden = [tf.nn.dropout(tf.nn.relu(tf.matmul(input_layer, W[0]) + b[0]), keep_prob)]
        for i in range(1, len(self.hidden_layers)):
            W.append(create_variable([self.hidden_layers[i-1], self.hidden_layers[i]]))
            b.append(create_variable([self.hidden_layers[i]]) + 0.5)
            hidden.append(tf.nn.dropout(tf.nn.relu(tf.matmul(hidden[-1], W[-1]) + b[-1]), keep_prob))
       
        W.append(create_variable([self.hidden_layers[-1], self.n_types]))
        b.append(create_variable([self.n_types]))
        
        y_pred = tf.nn.softmax(tf.matmul(hidden[-1], W[-1]) + b[-1])
        y_true = tf.one_hot(indices=type_batch, depth=self.n_types, on_value=1.0, off_value=0.0, dtype=tf.float32)
        
        loglike = tf.reduce_mean(tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1])) 
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(-loglike)

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
                        'p_throws', 'inning_half', 'batter_team', 'pitcher_team', 'type']
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
        type_batch = self.network['type_batch']
        train_step = self.network['train_step']
        keep_prob = self.network['keep_prob']
        sess = self.network['sess']

#        bar = progressbar.ProgressBar(maxval=self.sweeps, \
#                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#        bar.start()
    
        for i in range(self.sweeps):
#            bar.update(i+1)
            perm = np.random.permutation(self.pitches.shape[0])
            cat_data = self.cat_data[perm]
            real_data = self.real_data[perm]
            for start in range(0, len(perm) - self.batch_size, self.batch_size):
                end = start + self.batch_size
                input_data = {  cat_batch : cat_data[start:end,:-1], 
                                real_batch : real_data[start:end],  
                                type_batch : cat_data[start:end,-1],
                                keep_prob : 1 - self.dropout }
                sess.run(train_step, feed_dict = input_data)
#        bar.finish()

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

