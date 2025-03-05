import numpy as np
import os.path
import pickle
import ucimlrepo as uci
from collections import defaultdict
from mdlp.discretization import MDLP
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# NOTE: hack to get outdated MDLP to work
np.bool = bool
data_dict = {}
encoders = {}
RANDOM_STATE = 123

def encode_transform(id, data):
    fit = np.zeros(data.shape)

    variables = data_dict[id].variables
    # Avoid pandas indexing
    variable_types = variables[variables.role == 'Feature'].type.to_numpy()

    for i in range(data.shape[1]):
        if i == data.shape[1]-1 or variable_types[i] in ['Categorical', 'Binary']:
            fit[:, i] = encoders[id][i].transform(data[:, i])
        else:
            fit[:, i] = data[:, i]

    return fit


def encode_invtransform(id, data):
    fit = np.empty(data.shape, dtype=object)

    variables = data_dict[id].variables
    # Avoid pandas indexing
    variable_types = variables[variables.role == 'Feature'].type.to_numpy()

    for i in range(data.shape[1]):
        if i == data.shape[1]-1 or variable_types[i] in ['Categorical', 'Binary']:
            fit[:, i] = encoders[id][i].inverse_transform(data[:, i].astype(int))
        else:
            fit[:, i] = data[:, i]

    return fit


def encode_transform_new(id, data):
    fit = np.zeros(data.shape)
    encoders[id] = defaultdict(LabelEncoder)

    variables = data_dict[id].variables
    # Avoid pandas indexing
    variable_types = variables[variables.role == 'Feature'].type.to_numpy()

    for i in range(data.shape[1]):
        if i == data.shape[1]-1 or variable_types[i] in ['Categorical', 'Binary']:
            fit[:, i] = encoders[id][i].fit_transform(data[:, i])
        elif variable_types[i] == 'Integer' and len(np.unique(data[:, i])) < 10:
            fit[:, i] = encoders[id][i].fit_transform(data[:, i])
        else:
            fit[:, i] = data[:, i]

    return fit


# TODO: faster to fork and only pass continuous features
def discretize(id, data):
    variables = data_dict[id].variables
    # Avoid pandas indexing
    variable_types = variables[variables.role == 'Feature'].type.to_numpy()
    for j in range(data.shape[1]-1):
        if variable_types[j] == 'Integer':
            if len(np.unique(data[:, j])) < 10:
                variable_types[j] = 'Discrete'
    continuous_features = [i for i, v in enumerate(variable_types) if v in ['Continuous', 'Integer']]

    if len(continuous_features) == 0:
        return data, None

    discretizer = MDLP(continuous_features=continuous_features, random_state=123)

    X, y = data[:, :-1], data[:, -1]

    X_disc = discretizer.fit_transform(X, y)
    data[:, :-1] = X_disc

    return (data, discretizer) # Just making it explicit that data is being mutated


def load_uci(id):
    filename = f"data/uci_pickle/{id}_data.pickle"

    # TODO: checksum
    if os.path.isfile(filename):
        with open(f"data/uci_pickle/{id}_data.pickle", "rb") as f:
            data = pickle.load(f)
    else:
        os.makedirs(os.path.dirname('data/uci_pickle'), exist_ok=True)
        data = uci.fetch_ucirepo(id=id)
        with open(f"data/uci_pickle/{id}_data.pickle", "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data


def preprocess(id, n_splits=2, n_repeats=1, load=True, save=True):
    if n_repeats > 1:
        raise NotImplementedError
    if load:
        try:
            with open(f"data/uci_pickle/{id}-mdlp-cv-{n_splits}.pickle", "rb") as f:
                splits = pickle.load(f)
            return splits
        except Exception:
            pass

    # Load UCI
    data_dict[id] = load_uci(id)
    data = data_dict[id].data.features.join(data_dict[id].data.targets).to_numpy()
    p = data.shape[1]
    data = encode_transform_new(id, data)
    cards = np.array([len(np.unique(data[:, i])) for i in range(p)])

    splits = []

    #if n_splits > 1:
    kf = KFold(n_splits=n_splits)
    iterable = kf.split(data)
    #else:
    #    train, test = train_test_split(data)
    #    iterable = [(train, test)]
    for train_idxs, test_idxs in iterable:
        train = data[train_idxs].copy()
        test = data[test_idxs].copy()
        # TODO: handle nans 
        train, disc = discretize(id, train)
        train = train.astype(int)
        disc_cards = cards

        # Update cards
        if disc is not None:
            for feature_idx in disc.continuous_features:
                disc_cards[feature_idx] = len(disc.cut_points_[feature_idx]) + 1

        # Preprocessing for validation set
        if disc: test[:, :-1] = disc.transform(test[:, :-1]).astype(int)

        # TODO: all X cards 
        train = train[: , disc_cards > 1]
        test = test[:, disc_cards > 1]
        disc_cards = disc_cards[disc_cards > 1]

        train = train.astype(int)
        test = test.astype(int)
        disc_cards = disc_cards.astype(int)
        splits.append((train, test, disc_cards))

    # Drop variables with cardinality 1

    if save:
        with open(f"data/uci_pickle/{id}-mdlp-cv-{n_splits}.pickle", "wb") as f:
            pickle.dump(splits, f, pickle.HIGHEST_PROTOCOL)

    return splits


def load_processed(id):
    with open(f"data/uci_pickle/{id}_mdlp.pickle", "rb") as f:
        datad = pickle.load(f)
    return datad
