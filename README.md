# Contextual Pitch Prediction 

This aim of this project is to predict the (distribution over) pitch types and locations of pitches based on contextual information such as the batter, the pitcher, the current count, player characteristics such as handedness and height, and other possibly relevant information.  

We develop models to characterize P_C(Type) and P_C(Location | Type) or the context dependent probability distributions of pitch type and pitch location given type.  With these two things together, we can recover the full join (context-dependent) probability distributions P_C(Type, Location) = P_C(Type) P_C(Location | Type).  This is a challenging problem because the number of training examples for a particular context decays exponentially with the number of context features we include in our models.  This is addressed by constructing models that automatically find the most relevant contextual features so that the models can generalize to new contexts and contexts without much training data.

Currently, we aim to create at least 3 models:

Model 1: The Baseline
Assume P_C(Type) and P_C(Location | Type) only depends on the pitcher (the other contextual information is irrelevant).  It is well understood how to model these two distributions, and there is enoughd data for most pitchers to construct these models. e) P_C(Type) can be modeled as a Categorical distribution while P_C(Location | Type) can be modeled approximately as a Gaussian Mixture Model.  

Model 2: Feed Forward Neural Networks
Modeling P_C(Type) easy to do with a neural network.  We model P_C(Location | Type) with a Mixture Density Network, or a feed forward neural network where the output neurons are used as parameters to a gaussian mixture model.  

Model 3: Recurrent Neural Network
Leverage additional temporal information such as past pitches using a recurrent neural network.

More information can be found in [here](https://docs.google.com/presentation/d/1AwiTHp89OIioVCS9c_Y6mDCBYJC9exFepivH6TePBBk/edit?usp=sharing).
