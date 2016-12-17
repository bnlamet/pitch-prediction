import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_curve(labels: list, probas: np.array, actuals: np.array, buckets: int):
    """
    Parameters
    ----------
    lables: list of length d
        Contains a label for each pitch type.
    probas: An d by N numpy matrix
        Contains probability estimates. probas[i, j] is the probability that experiment i has label j.
        Each column must sum to 1.
    actuals: vector of length N
        The ith value is the (index in `labels` of the) observed pitch type.
    buckets: int
        Number of buckets to partition the data into.
    """
    d = len(labels)
    shape = np.shape(probas)
    N = shape[1]

    probas = np.asarray(probas)
    actuals = np.asarray(actuals)
    
    assert len(np.shape(labels)) == 1, 'labels should be 1D'
    assert shape[0] == d, 'probas should have d rows'
    assert len(np.shape(actuals)) == 1 and len(actuals) == N, 'actuals should be a 1D vector of length N'
    assert probas.min() >= 0 and probas.max() <= 1, 'probas should be real probabilities'

    fig = plt.figure()
    axis = fig.add_axes((0, 0, 1, 1))
    axis.plot([0,1], [0,1], 'k:', label='Perfectly calibrated')
    axis.set_ylabel("Fraction of positives")
    axis.set_ylim([-0.05, 1.05])
    axis.legend(loc="lower right")
    axis.set_title('Calibration plots (reliability curve)')
    for i in range(0, 3):
        y_prob = probas[i]
        y_true = (actuals == i)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=buckets)
#        axis.plot(prob_true, prob_pred, "s-", label=labels[i])
        axis.plot(prob_true, prob_pred, ".", label=labels[i])

    return fig
