# Hierarchical Linear Smoothing
**Efficient Parameter Estimation for Bayesian Network Classifiers using Hierarchical Linear Smoothing**

TODO *abstract*

TODO *citation*

## Code

TODO describe difference between ridge and bayesian
TODO if using bayesian sampler, pgdraw will be compiled on import
TODO or alternatively to compile the Cython code for the Polya-Gamma sampler, run the following under the `pgdraw/` directory:
```
python setup.py build_ext --inplace
```

### Dependencies

Requires poetry.

Dependencies can be installed and environment activated using `poetry`:
```
poetry install
poetry shell
```

### Example

`hls_functions.py`

TODO permalink example.py

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
