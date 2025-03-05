import numpy as np
import operator
import scipy.stats as st
import scipy.special as sp
import timeit
from itertools import accumulate, product
from scipy.sparse import csc_matrix
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from utils import design_matrix, expanded
from HDP import pyHDP

# TODO: rewrite
def generate_correlated_categorical_matrix(n_samples, n_features, cardinality, correlation_mean, correlation_variance, random_state=None):
    np.random.seed(random_state)

    # generate random correlation matrix
    base_corr = np.eye(n_features)
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr = np.random.normal(correlation_mean, np.sqrt(correlation_variance))
            corr = np.clip(corr, 0, 1)  # Ensure correlation is between 0 and 1
            base_corr[i, j] = corr
            base_corr[j, i] = corr

    # ensure positive semi-definite
    min_eigenvalue = np.min(np.linalg.eigvals(base_corr))
    if min_eigenvalue < 0:
        base_corr += (-min_eigenvalue + 1e-6) * np.eye(n_features)

    # normalize 
    D = np.diag(1 / np.sqrt(np.diag(base_corr)))
    base_corr = D @ base_corr @ D

    # generate correlated normal variables
    latent = np.random.multivariate_normal(np.zeros(n_features), base_corr, size=n_samples)
    # transform to uniform distribution
    uniform = norm.cdf(latent)
    # transform to categorical
    data = np.floor(uniform * cardinality).astype(int)
    # ensure the categories are within [0, cardinality-1]
    data = np.clip(data, 0, cardinality - 1)

    return data, base_corr

from utils import eye_squished
def design_matrix(cardinalities, alphas=None):
    db = [1] + list(accumulate(cardinalities[:-1], operator.mul))
    #dbr = list(reversed([1] + list(accumulate(reversed(cards[:-1]), operator.mul))))
    Xs = list(map(lambda i: eye_squished(db[-1], db[i]), reversed(range(len(cardinalities)))))
    if alphas is not None:
        for i in range(len(alphas)):
            Xs[i] *= alphas[i]
    strings = np.block(Xs)
    return strings

# Example usage
cardinality = 5
correlation_mean = 0.5
correlation_variance = 0.04  # This gives a standard deviation of 0.2

gateway = pyHDP.init_gateway()

def hdp_time(matrix):
    tree = pyHDP.fit_hdp(matrix, gateway)

def hls_time(sparse, design_exp, counts_exp):
    sparse_exp = csc_matrix(design_exp)
    clf = LogisticRegression(
        C=1.0,
        solver='newton-cg',
        penalty='l2',
        fit_intercept=False).fit(sparse_exp, counts_exp)
    beta = clf.coef_.T
    eta = sparse @ beta
    if counts.shape[1] == 2:  # binomial
        mu = sp.expit(eta)
    else:
        mu = sp.softmax(eta, axis=1)

def hls_time_with_exp(y):
    design = design_matrix(cards)[:, :-1]
    counts = np.c_[np.unique(y, return_counts=True)[1]].T
    design_exp, counts_exp = expanded(design, counts)
    sparse = csc_matrix(design)
    sparse_exp = csc_matrix(design_exp)
    clf = LogisticRegression(
        C=1.0,
        solver='newton-cg',
        penalty='l2',
        fit_intercept=False).fit(sparse_exp, counts_exp)
    beta = clf.coef_.T
    eta = sparse @ beta
    if counts.shape[1] == 2:  # binomial
        mu = sp.expit(eta)
    else:
        mu = sp.softmax(eta, axis=1)

num_samples = [100, 1000, 10000, 100000]
num_features = [2, 3, 4, 5]
cardinalities = [2, 3, 4, 5]
num_samples = [100, 1000, 10000, 100000]
num_features = [2, 5]
cardinalities = [2, 5]
for N, p, c in product(num_samples, num_features, cardinalities):
    matrix, corr_matrix = generate_correlated_categorical_matrix(
        N, p+1, c, correlation_mean, correlation_variance, random_state=123
    )
    X = matrix[:, :-1]
    y = matrix[:, -1]
    cards = np.full(p, c)
    design = design_matrix(cards)[:, :-1]
    counts = np.c_[np.unique(y, return_counts=True)[1]].T
    design_exp, counts_exp = expanded(design, counts)
    sparse = csc_matrix(design)

    time1 = timeit.repeat(
        lambda: hls_time(sparse, design_exp, counts_exp),
        repeat=10, number=1)
    time2 = timeit.repeat(
        lambda: hls_time_with_exp(y),
        repeat=10, number=1)
    time3 = timeit.repeat(
        lambda: hdp_time(matrix),
        repeat=10, number=1)

    print(f"num_samples = {N}, num_features = {p}, card = {c}")
    print(f"hls:        {min(time1)}")
    print(f"hls w/ exp: {min(time2)}")
    print(f"hdp:      : {min(time3)}")
    print()
