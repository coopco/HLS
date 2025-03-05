import operator
import numpy as np
from itertools import accumulate, product
from scipy.special import expit, softmax
from sklearn.metrics import mean_squared_error, zero_one_loss, log_loss


def eye_squished(m, n):
    """
    Create a 'squished' identity matrix

    Parameters:
        m (int): number of rows
        n (int): number of columns
    """
    return np.array([[j == np.ceil(i*n/m)
        for j in range(1, n+1)]
        for i in range(1, m+1)],
                    dtype=int)


# TODO: consistent cardinalities include y or not
def data_matrix(cardinalities):
    return np.array(list(product(*[range(card) for card in cardinalities])))


# TODO: make sparse-version with time linear in number of non-zeros
def design_matrix(cardinalities):
    db = [1] + list(accumulate(cardinalities[:-1], operator.mul))
    #dbr = list(reversed([1] + list(accumulate(reversed(cards[:-1]), operator.mul))))
    Xs = list(map(lambda i: eye_squished(db[-1], db[i]), reversed(range(len(cardinalities)))))
    strings = np.block(Xs)
    return strings


def expanded(X, y, n=None):
    if len(y.shape) == 1:
        y = np.c_[n-y, y]
    J = y.shape[1]

    x_full = np.r_[
        *[np.repeat([X[i]], count, axis=0) for j in range(J) for i, count in enumerate(y[:, j])]
    ]

    y_full = np.r_[
        *[np.repeat([j], count, axis=0) for j in range(J) for i, count in enumerate(y[:, j])]
    ]

    if n is not None:
        return x_full, y_full, np.ones(x_full.shape[0])
    else:
        return x_full, y_full


def marginal_prob(X):
    """Returns marginal probability distribution of X"""
    marginals = {x: len(X[X == x]) / len(X) for x in np.unique(X)}
    return marginals


def joint_prob_2(X: np.array, Y: np.array):
    """Returns joint probability distribution of X and Y"""
    joints = {(x, y): len(X[(X == x) & (Y == y)]) / len(X)
              for x in np.unique(X)
              for y in np.unique(Y)}
    return joints


def joint_prob_3(X, Y, Z):
    """Returns joint probability distribution of X, Y, and Z"""
    joints = {(x, y, z): len(X[(X == x) & (Y == y) & (Z == z)]) / len(X)
              for x in np.unique(X)
              for y in np.unique(Y)
              for z in np.unique(Z)}
    return joints


def joint_prob(*Xs):
    """Returns joint probability distribution of arguments"""
    if len(Xs) == 2:
        return joint_prob_2(*Xs)
    elif len(Xs) == 3:
        return joint_prob_3(*Xs)
    else:
        raise NotImplementedError


def mutual_information(X, Y):
    """Computes MI(X, Y)"""
    marginal_Xs = marginal_prob(X)
    marginal_Ys = marginal_prob(Y)
    joint_XYs = joint_prob(X, Y)
    mi = sum([joint_XYs[x, y] * np.log(
        joint_XYs[x, y] /
        (marginal_Xs[x]*marginal_Ys[y]))
        for x in np.unique(X)
        for y in np.unique(Y)
        if joint_XYs[x, y] != 0])
    return mi


def conditional_mutual_information(X, Y, Z):
    """Computes MI(X, Y; Z)"""
    marginal_Zs = marginal_prob(Z)
    joint_XZs = joint_prob(X, Z)
    joint_YZs = joint_prob(Y, Z)
    joint_XYZs = joint_prob(X, Y, Z)
    mi = sum([joint_XYZs[x, y, z] * np.log(
        marginal_Zs[z]*joint_XYZs[x, y, z] /
        (joint_XZs[x, z]*joint_YZs[y, z]))
        for x in np.unique(X)
        for y in np.unique(Y)
        for z in np.unique(Z)
        if joint_XYZs[x, y, z] != 0])

    return mi

###
### Losses
###

def zero_one_loss_classification(predictions, targets):
    """
    predictions:
    targets: 
    """
    return zero_one_loss(targets, np.argmax(predictions, axis=1))


def rmse_classification(predictions, targets):
    """
    predictions:
    targets: 
    """
    # One hot
    y = np.zeros(predictions.shape)
    for row in range(predictions.shape[0]):
        y[row][targets[row]] = 1

    return np.sqrt(mean_squared_error(y, predictions))


def log_loss_classification(predictions, targets, labels=None):
    if labels == None:
        return log_loss(targets, predictions)
    else:
        return log_loss(targets, predictions, labels=labels)


def sample_mus(X, samples):
    if len(samples['beta'].shape) > 2:
        eta_samples = X @ samples['beta'] + samples['beta_0'][:, None, :]
        mu_samples = softmax(eta_samples, axis=2)
        mu = mu_samples.mean(axis=0)
    else:
        eta_samples = X @ samples['beta'].T + samples['beta_0']
        mu_samples = expit(eta_samples)
        mu = mu_samples.mean(axis=1)
    return mu
