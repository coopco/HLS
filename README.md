# Hierarchical Linear Smoothing
**Efficient Parameter Estimation for Bayesian Network Classifiers using Hierarchical Linear Smoothing**

> Bayesian network classifiers (BNCs) possess a number of properties desirable for a modern classifier: They are easily interpretable, highly scalable, and offer adaptable complexity. However, traditional methods for learning BNCs have historically underperformed when compared to leading classification methods such as random forests. Recent parameter smoothing techniques using hierarchical Dirichlet processes have enabled BNCs to achieve performance competitive with random forests on categorical data, but these techniques are relatively inflexible, and require a complicated, specialized sampling process. In this paper, we introduce a novel method for parameter estimation that uses a log-linear regression to approximate the behaviour of hierarchical Dirichlet processes. As a linear model, our method is remarkably flexible and simple to interpret, and can leverage the vast literature on learning linear models. Our extensive experiments show that our method achieves accuracy comparable to state-of-the-art BNCs and random forests on categorical data, while being orders of magnitude faster than the former and supporting direct explainability in contrast to the latter.

TODO *bibtex citation*

## Code

The main HLS functions are provided in `hls_functions.py`. Structure learning methods are provided in `bnc.py`
HLS can be used with a simple sklearn ridge regression, or with a Bayesian ridge implemented in `hls_sampler.py`


### Dependencies

Requires poetry.

Dependencies (for HLS-NB) can be installed and environment activated using `poetry`:
```
poetry install
poetry shell
```
If using the Bayesian HLS (HLS-IG), some additional dependencies must be installed using
```
poetry install --with glsh
```
Cython code under `pgdraw/` will be compiled when imported for the first time. 
Alternatively, you can compile the pgdraw code for the by running the following under the `pgdraw/` directory:
```
python setup.py build_ext --inplace
```


### Example (in `example.py`)

A basic example using Iris data:
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Any discretization method can be used, here we use a basic example:
```python
from sklearn.preprocessing import KBinsDiscretizer

# basic discretizer
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal')
discretizer.fit(X_train)  # fit to training set

# transform, and ensure ints
X_train_discretized = discretizer.transform(X_train).astype(int)
X_test_discretized = discretizer.transform(X_test).astype(int)

# need to calculate the cardinalities of each feature
cards = np.r_[np.max(X_train_discretized, axis=0), np.max(y)].astype(int) + 1
```

For the structure learning step:
```python
from bnc import tan, kdb

# train structure
edges = tan(X_train, y_train)  # or use kdb(X_tran, y_train, k=3)
```

Finally, the parameters can be learned as follows:
```python
from hls_functions import train_hls_bnc
from bnc import probs_to_pgmpy, pgmpy_infer

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
```

## Reproducing experiments


### Dependencies
Experiments require additional dependencies which can be installed using `poetry`:
```
poetry install --with experiments
poetry shell
```
Experiments use mdlp-discretization which might need to be installed separately, once in the poetry shell:
```
pip install git+https://github.com/hlin117/mdlp-discretization
```

### Running the experiments

Experiments are described in yaml files.
When running experiments, the UCI datasets will be downloaded, preprocessed, and saved to `data/uci_pickle`.
Results are saved to `results/`, and experiments are skipped if there are already existing results under that directory.

To run the experiments for the main paper:
```
python experiments.py yaml/main.yaml
```

To run the experiments for appendices:
```
python experiments.py yaml/supplementary.yaml
```

To run the scaling experiment on random data:
```
python scaling.py
```
Results are printed to stdout.

TODO: the additional scaling experiments

### HDP experiments
To run the HDP experiments, you will need to have the jdk installed and run the following command in the `HDP/` directory:
```
javac -cp .:py4j0.10.9.7.jar:dist/HDP.jar:lib/*:lib/commons-math3-3.6.1/* pyHDP
```
Then you can start a py4j gateway under `HDP/`
```
java -Xmx8g -cp .:lib/*:lib/HDP/*:lib/commons-math3-3.6.1/* pyHDP
```
and run
```
python experiments.py yaml/hdp.yaml
```
The program might exit with an error due to the JVM running out of ram, in which case you must restart the gateway and the experiments (which will skip experiments already completed).
