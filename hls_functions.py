import numpy as np
import operator
import scipy.special as sp
from itertools import accumulate, product
from tqdm import tqdm
from scipy.sparse import csc_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

import utils
from bnc import edges_to_parents
from trie import Trie

from utils import sample_mus
from hls_sampler import sampler


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


# TODO: make sparse-version with time linear in number of non-zeros
def design_matrix(cardinalities):
    db = [1] + list(accumulate(cardinalities[:-1], operator.mul))
    Xs = list(map(lambda i: eye_squished(db[-1], db[i]), reversed(range(len(cardinalities)))))
    strings = np.block(Xs)
    return strings


def counts_trie(data, child, parents, cards):
    target_idx = child
    parent_idxs = parents[child]

    counts_trie = Trie(cards[parent_idxs + [target_idx]])
    counts_trie.insert_set(data, parent_idxs + [target_idx])

    p = len(parent_idxs)+1
    # NOTE: Relies on edges being sorted on mutual information
    c = cards[parent_idxs + [target_idx]]
    db = [1] + list(accumulate(c[:-1], operator.mul))
    counts = np.zeros((db[-1], c[-1]), dtype=int)

    # TODO: move this into trie method
    # Turn trie into matrix
    for i, string in enumerate(product(*[range(x) for x in c[:-1]])):
        result = counts_trie.search(string)
        counts[i, :] = result

    # TODO: somewhere else
    counts = counts.astype(float)
    return counts


def train_hls(counts, c, version="ridge", ridge_lambda=1.0):
    assert version in ["ridge", "bayesian"]

    # Handle classes with zero counts
    zero_counts = counts.sum(0) == 0
    nonzero_counts = counts[:, np.invert(zero_counts)]

    strings = design_matrix(c)[:, :-1]  # Drop the column of 1s
    strings_exp, nonzero_counts_exp = utils.expanded(strings, nonzero_counts)
    strings_sparse = csc_matrix(strings)
    strings_sparse_exp = csc_matrix(strings_exp)

    if version == "bayesian":
        prior = "ridge"
        n_burn = 2000
        n_samples = 2000
        samples = sampler(
            strings, counts, prior,
            n_burn=n_burn, n_samples=n_samples,
            scale_prior="inverse-gamma")
        mu = sample_mus(strings, samples)
        return mu
    elif version == "ridge":
        clf_ridge = LogisticRegression(
            C=ridge_lambda,
            solver='newton-cg',
            penalty='l2',
            fit_intercept=False).fit(strings_sparse_exp, nonzero_counts_exp)
        clf_beta = clf_ridge.coef_.T
        clf_beta_0 = clf_ridge.intercept_

        # Handle zero_counts
        if counts.shape[1] == 2:
            beta = clf_beta
            beta_0 = clf_beta_0
        else:
            beta = np.zeros((strings.shape[1], counts.shape[1]))
            beta_0 = np.zeros(counts.shape[1])
            beta[:, np.invert(zero_counts)] = clf_beta
            beta_0[np.invert(zero_counts)] = clf_beta_0

        eta = strings @ beta + beta_0
        if counts.shape[1] == 2:  # binomial
            mu = sp.expit(eta)
            mu = np.c_[1-mu, mu]  # Tw
        else:  # multinomial
            mu = sp.softmax(eta, axis=1)

        return mu


# TODO: neater
def compute_cards(X, y):
    train = np.c_[X, y]
    p = np.shape(train)[1]
    cards = np.array([len(np.unique(train[:, i])) for i in range(p)])
    return cards

#X = train[:, :-1]
#y = train[:, -1]
def train_hls_bnc(X, y, cards, edges, version="ridge"):
    """
    """
    train = np.c_[X, y]
    p = np.shape(train)[1]

    parents = edges_to_parents(edges, X.shape[1])

    # Holds TODO:
    node_probs = [np.array([]) for _ in range(len(parents))]

    # train nodes
    for child in tqdm(range(len(parents))):
        counts = counts_trie(train, child, parents, cards)

        target_idx = child
        parent_idxs = parents[child]
        c = cards[parent_idxs + [target_idx]]
        mu = train_hls(counts, c, version)
        node_probs[child] = mu

    # Create pgmpy network
    class_counts = np.zeros(cards[-1])
    for yi in y:
        class_counts[yi] += 1
    class_probs = normalize(np.expand_dims(class_counts, 0), axis=1, norm="l1")

    node_probs.append(class_probs)
    return node_probs

