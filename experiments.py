import numpy as np
import scipy.special as sp
import os
import operator
import typer
import yaml
from dataclasses import dataclass
from itertools import accumulate, product
from scipy.sparse import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import normalize
from tqdm import tqdm

import utils
from bnc import tan, kdb, edges_to_parents, pgmpy_infer, probs_to_pgmpy
from HDP import pyHDP
from hls_sampler import sampler as multi_sampler
from trie import Trie
from uci import preprocess
from utils import design_matrix, expanded, sample_mus, data_matrix

EXPERIMENTS_DIR = "results"

UCI_IDS = [12, 17, 19, 30, 42, 43, 44, 50, 53, 63, 69, 70, 76, 78, 94, 95, 96,
           101, 107, 109, 111, 145, 174, 176, 186, 212, 257, 267, 277, 292,
           327, 329, 372, 379, 445, 519, 529, 545, 563, 572, 582, 697, 759,
           827, 850, 856, 863, 890, 915, 936]

BAD_HDP_IDS = [69, 78, 697, 827, 856, 863, 915, 936]

RANDOM_STATE = 123
np.random.seed(123)

app = typer.Typer(pretty_exceptions_enable=False)

# TODO: just directly convert dictionary
@dataclass(frozen=True)
class ExperimentParams:
    name: str
    structure: str
    kdb_k: int = 3
    pseudo_counts: int = 0
    solver: str = ""
    scipy_ridge: bool = True
    fit_intercept: bool = False
    n_splits: int = 10
    n_repeats: int = 1
    fixed_tau: float | None = None
    backoff: bool = True
    ridge_c: float = 1  # Ridge inverse regularization strength
    scale_prior: str = "cauchy"  # cauchy or 'inverse-gamma'
    ghs_a: float = 0.5  # Beta(a, b) in the generalised horseshoe
    ghs_b: float = 0.5
    global_scale: bool = True  # Scale parameter for all categories
    group_scale: bool = False  # Each category has its own scale parameter
    one_off: str = ""  # For once-off experiments, give a different filename


def experiment_filename(experiment_params):
    m = experiment_params
    filename = f'{EXPERIMENTS_DIR}/{m.name}' + \
    f'-{m.solver}'*(m.solver != "") + \
    f'-{m.structure}' + \
    f'-{m.kdb_k}'*(m.structure == "kdb") + \
    f'-{m.scale_prior}'*(m.scale_prior != "cauchy") + \
    f'-ghs_a={m.ghs_a}'*(m.ghs_a != 0.5) + \
    f'-ghs_b={m.ghs_b}'*(m.ghs_b != 0.5) + \
    '-no_tau'*(m.global_scale is False) + \
    '-vu'*(m.group_scale is True) + \
    f'-add{m.pseudo_counts}'*(m.pseudo_counts > 0) + \
    '-intercept'*(m.fit_intercept) + \
    f'-fixed{m.fixed_tau}'*(m.fixed_tau is not None) + \
    f'-cv-{m.n_splits}' + f'-{m.n_repeats}'*(m.n_repeats > 1) + \
    f'-{m.one_off}'*(m.one_off != "") + \
    '.csv'
    return filename


def run_uci(params):
    ids = UCI_IDS
    filename = experiment_filename(params)
    print(filename)
    file_exists = os.path.isfile(filename)
    file_empty = file_exists and os.stat(filename).st_size == 0
    curr_ids = []
    if file_exists and not file_empty:
        curr_results = np.genfromtxt(filename, delimiter=',', skip_header=1)
        if curr_results.shape != (0,):
            curr_ids = curr_results[:, 0]

    for id in tqdm(ids):
        if params.name == "hdp" and id in BAD_HDP_IDS:
            continue
        if id in curr_ids:
            cv_steps = curr_results[:, 1][curr_results[:, 0] == id].astype(int)
            if len(cv_steps) == 10: continue
            train_cv(params, id, skip=cv_steps)
            continue

        train_cv(params, id)

# for split in n_splits
def train_cv(params, id, skip=[]):
    n_splits = params.n_splits
    n_repeats = params.n_repeats

    # aggregate statistics
    train_rmses = np.zeros(n_splits*n_repeats)
    train_01s = np.zeros(n_splits*n_repeats)
    train_logs = np.zeros(n_splits*n_repeats)
    test_rmses = np.zeros(n_splits*n_repeats)
    test_01s = np.zeros(n_splits*n_repeats)
    test_logs = np.zeros(n_splits*n_repeats)
    end_times = np.zeros(n_splits*n_repeats)

    splits = preprocess(id, n_splits=n_splits, n_repeats=n_repeats)

    os.makedirs(EXPERIMENTS_DIR)

    for split in range(n_splits*n_repeats):
        if split in skip:
            continue
        # TODO: n_repeats > 1
        train, test, cards = splits[split]
        test_x = test[:, :-1]
        test_y = test[:, -1]
        X = train[:, :-1]
        y = train[:, -1]

        # train model
        # if bayesian network
        if params.name in ["random-forest"]:
            train_pred, test_pred = train_rf(params, cards, X, y, test_x, test_y)
        else:
            bnc = train_bnc(params, train, cards)
            train_pred = pgmpy_infer(bnc, X, cards)
            test_pred = pgmpy_infer(bnc, test_x, cards)

        class_labels = list(range(cards[-1]))
        train_losses = losses(y, train_pred, class_labels)
        test_losses = losses(test_y, test_pred, class_labels)
        train_rmses[split], train_01s[split], train_logs[split] = train_losses
        test_rmses[split], test_01s[split], test_logs[split] = test_losses

        # Save row
        train_rmse = train_rmses[split]
        train_01 = train_01s[split]
        train_log = train_logs[split]
        test_rmse = test_rmses[split]
        test_01 = test_01s[split]
        test_log = test_logs[split]
        end_time = 0
        # TODO: save outside this method
        filename = experiment_filename(params)
        with open(filename, 'a') as f:
            f.write(f"\n{id}, {split}, {train_rmse}, {train_01}, {train_log}, {test_rmse}, {test_01}, {test_log}, {end_time}")

def train_bnc(params, train, cards):
    p = params
    X = train[:, :-1]
    y = train[:, -1]

    # learn structure
    print("Learning structure...")
    if p.structure == "tan":
        edges = tan(X, y)
    else:
        edges = kdb(X, y, k=p.kdb_k)
    parents = edges_to_parents(edges, X.shape[1])
    node_samples = [np.array([]) for _ in range(len(parents))]
    node_probs = [np.array([]) for _ in range(len(parents))]

    # train nodes
    for child in tqdm(range(len(parents))):
        if len(parents[child]) == 0:
            breakpoint()
            child_counts = np.zeros(cards[child])
            for xi in train:
                child_counts[xi[child]] += 1
            child_counts += p.pseudo_counts

            child_probs = np.expand_dims(child_counts / X.shape[0], 0)
            child_probs = normalize(np.expand_dims(child_counts, 0), axis=1, norm="l1")
            node_probs[child] = child_probs

        counts = counts_trie(train, child, parents, cards)

        target_idx = child
        parent_idxs = parents[child]
        c = cards[parent_idxs + [target_idx]]
        if p.name in ["hdp"]:
            if p.pseudo_counts > 0:
                raise NotImplementedError
            mu = train_hdp(params, train, counts.astype(int), c,
                           child, parents, target_idx)
        elif p.name in ["additive-smoothing"]:
            # TODO: avoid counting trie twice
            trie = Trie(cards[parent_idxs + [target_idx]])
            trie.insert_set(train, parent_idxs + [target_idx])
            if p.backoff is False:
                # Add counts before backoff
                counts += p.pseudo_counts
            mu = train_dirichlet(params, counts, c, trie)
        else:
            counts += p.pseudo_counts
            mu = train_hls(params, counts, c)
        node_probs[child] = mu

    # Creat pgmpy network
    class_counts = np.zeros(cards[-1])
    for yi in y:
        class_counts[yi] += 1
    class_counts += p.pseudo_counts
    class_probs = normalize(np.expand_dims(class_counts, 0), axis=1, norm="l1")

    node_probs.append(class_probs)
    try:
        bnc = probs_to_pgmpy(node_probs, edges, len(cards), cards)
    except ValueError:
        breakpoint()
    return bnc

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


def train_hls(params, counts, c):
    p = params

    # Handle classes with zero counts
    zero_counts = counts.sum(0) == 0
    nonzero_counts = counts[:, np.invert(zero_counts)]

    strings = design_matrix(c)[:, :-1]  # Drop the column of 1s
    strings_exp, nonzero_counts_exp = expanded(strings, nonzero_counts)
    strings_sparse = csc_matrix(strings)
    strings_sparse_exp = csc_matrix(strings_exp)

    clf_beta = []
    clf_beta_0 = []

    if p.name in ["horseshoe", "bayes-ridge"] and p.solver == "glsh":
        prior = "horseshoe" if p.name == "horseshoe" else "ridge"
        n_burn = 2000
        n_samples = 2000
        samples = multi_sampler(
            strings, counts, prior,
            fixed_tau=p.fixed_tau, fit_intercept=p.fit_intercept,
            n_burn=n_burn, n_samples=n_samples,
            scale_prior=p.scale_prior, a=p.ghs_a, b=p.ghs_b,
            global_scale=p.global_scale, group_scale=p.group_scale)
        mu = sample_mus(strings, samples)
        return mu
    elif p.name == "bayes-ridge" and p.solver == "scipy":
        raise NotImplementedError
    elif p.name in ["ridge", "lasso"]:
        clf_ridge = LogisticRegression(
            C=p.ridge_c,
            solver='newton-cg' if p.name == "ridge" else 'saga',
            penalty='l2' if p.name == "ridge" else 'l1',
            fit_intercept=p.fit_intercept).fit(strings_sparse_exp, nonzero_counts_exp)
        clf_beta = clf_ridge.coef_.T
        clf_beta_0 = clf_ridge.intercept_
    elif p.name == "ridge-cv":
        try:
            min_count = np.min(np.unique(nonzero_counts_exp, return_counts=True)[1])
            if min_count > 1:
                clf_ridge = LogisticRegressionCV(
                    #Cs=1 / ...,
                    solver='newton-cg',
                    cv=np.min([min_count, 5]),
                    penalty='l2',
                    fit_intercept=p.fit_intercept).fit(strings_sparse_exp, nonzero_counts_exp)
            else:
                clf_ridge = LogisticRegression(
                    C=p.ridge_c,
                    solver='newton-cg',
                    penalty='l2',
                    fit_intercept=p.fit_intercept).fit(strings_sparse_exp, nonzero_counts_exp)
        except Exception as e:
            print(e)
            breakpoint()
        clf_beta = clf_ridge.coef_.T
        clf_beta_0 = clf_ridge.intercept_
    else:
        raise NotImplementedError

    # Handle zero_counts
    if counts.shape[1] == 2:
        beta = clf_beta
        beta_0 = clf_beta_0
    else:
        try:
            beta = np.zeros((strings.shape[1], counts.shape[1]))
            beta_0 = np.zeros(counts.shape[1])
            beta[:, np.invert(zero_counts)] = clf_beta
            beta_0[np.invert(zero_counts)] = clf_beta_0
        except Exception:
            breakpoint()

    eta = strings @ beta + beta_0
    if counts.shape[1] == 2:  # binomial
        mu = sp.expit(eta)
        mu = np.c_[1-mu, mu]  # Tw
    else:  # multinomial
        mu = sp.softmax(eta, axis=1)

    return mu


def train_dirichlet(params, counts, c, trie):
    """Additive smoothing."""
    ml_counts = np.copy(counts)
    # List of configurations of parents
    configs = np.array(list(product(*[range(x) for x in c[:-1]])))
    # For each index of zero-counts
    for i in np.arange(ml_counts.shape[0])[ml_counts.sum(axis=1) == 0]:
        backoff_config = configs[i]
        count = ml_counts[i]
        # Backoff count until count > 0 is found
        while count.sum() == 0.:
            backoff_config = backoff_config[:-1]
            count = trie.search(backoff_config)
        ml_counts[i, :] = count

    if params.pseudo_counts == 0:
        # counts exactly 0 can result in problems
        ml_counts += 0.0001
    else:
        ml_counts += params.pseudo_counts
    mu = normalize(ml_counts, axis=1, norm="l1")
    return mu

# TODO: janky
def train_hdp(params, train, counts, c, child, parents, target_idx):
    gateway = pyHDP.init_gateway()
    try:
        tree = pyHDP.fit_hdp(train[:, parents[child] + [target_idx]], gateway)
        configs = data_matrix(c[:-1])
        mu = pyHDP.probs_hdp(tree, configs, gateway)
    except Exception as e:
        print(e)
        breakpoint()
    # If one of the classes had zero total counts
    if mu.shape[1] != c[-1]:
        zero_counts = counts.sum(0) == 0
        new_mu = np.zeros((mu.shape[0], counts.shape[1]))
        new_mu[:, zero_counts] += 1e-10
        new_mu[:, np.invert(zero_counts)] = mu
        mu = new_mu
    return mu

def random_forest(X, y):
    criterion = "gini" # {"gini", "entropy", "log_loss"}
    n_estimators = 100
    max_features = "sqrt"
    min_samples_leaf = 1

    rf = RandomForestClassifier(
        criterion=criterion,
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE
    )

    rf.fit(X, y)
    return rf

# TODO: cleaner
def train_rf(params, cards, X, y, test_x, test_y):
    if X.shape[1] == 0:
        probs = np.unique(y, return_counts=True)[1] / y.shape[0]
        rf_train_pred = np.repeat([probs], y.shape[0], axis=0)
        rf_test_pred = np.repeat([probs], test_y.shape[0], axis=0)
    else:
        rf = random_forest(X, y)
        rf_train_pred = rf.predict_proba(X)
        rf_test_pred = rf.predict_proba(test_x)

    # False for classes (indexes) that have zero counts
    has_counts = np.array([i in y for i in list(range(cards[-1]))])

    # If any class has zero samples
    if rf_train_pred.shape[1] < cards[-1]:
        train_pred = np.zeros((X.shape[0], cards[-1]))
        test_pred = np.zeros((test_x.shape[0], cards[-1]))
        train_pred[:, has_counts] = rf_train_pred
        test_pred[:, has_counts] = rf_test_pred
    else:
        train_pred = rf_train_pred
        test_pred = rf_test_pred

    return train_pred, test_pred

def losses(y, pred, class_labels):
    """Compute losses given predictions and targets"""
    rmse = utils.rmse_classification(pred, y)
    zeroone = utils.zero_one_loss_classification(pred, y)
    log = utils.log_loss_classification(pred, y, labels=class_labels)
    return rmse, zeroone, log

def run_experiment(params: ExperimentParams):
    typer.echo(params)
    run_uci(params)

def expand_experiment_params(params_dict):
    keys, values = zip(*params_dict.items())
    expanded_experiments = []

    for combination in product(*values):
        experiment = dict(zip(keys, combination))
        expanded_experiments.append(ExperimentParams(**experiment))

    return expanded_experiments

@app.command()
def load_and_run_experiments(config_file: str):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    experiments = config.get('experiments', [])

    for experiment in experiments:
        # cast all values to lists
        for key in experiment:
            if not isinstance(experiment[key], list):
                experiment[key] = [experiment[key]]

        expanded_experiments = expand_experiment_params(experiment)

        for exp in expanded_experiments:
            print(exp)
            run_experiment(exp)


if __name__ == "__main__":
    app()
