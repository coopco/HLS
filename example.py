import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from bnc import tan, kdb, probs_to_pgmpy, pgmpy_infer
from hls_functions import train_hls_bnc

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# basic discretizer
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal')
discretizer.fit(X_train)  # fit to training set

# transform
X_train_discretized = discretizer.transform(X_train).astype(int)
X_test_discretized = discretizer.transform(X_test).astype(int)

cards = np.r_[np.max(X_train_discretized, axis=0), np.max(y)].astype(int) + 1

# train structure
edges = tan(X_train, y_train)  # or use kdb(X_tran, y_train, k=3)

# learn parameters
node_probs = train_hls_bnc(X_train_discretized, y_train, cards, edges,
                           version="ridge")  # or use version="bayesian"
bnc = probs_to_pgmpy(node_probs, edges, len(cards), cards)

# predicted probabilities
train_probs = pgmpy_infer(bnc, X_train_discretized, cards)
test_probs = pgmpy_infer(bnc, X_test_discretized, cards)

# predicted classes
train_pred = np.argmax(train_probs, axis=1)
test_pred = np.argmax(test_probs, axis=1)

accuracy = accuracy_score(y_test, test_pred)
print(accuracy)
