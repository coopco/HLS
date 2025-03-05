import numpy as np
import utils as utils
from typing import List
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def naive_bayes(X, Y):
    """
    """
    num_col = X.shape[1]
    # Naive bayes
    edges = [(num_col, i) for i in range(num_col)]
    return edges


#  Zheng, F., Webb, G.I. (2011). Tree Augmented Naive Bayes.
#  https://doi.org/10.1007/978-0-387-30164-8_850
# Naive bayes + chow_liu tree (conditional mutual information instead)
#   Maximum spanning tree where weight is pairwise mutual information between each edge
# TODO: linear complexity algorithms exist since graph is dense
def tan(X, Y):
    num_col = X.shape[1]

    # Naive bayes
    edges = [(num_col, i) for i in range(num_col)]

    # Calculate MI(X_i; X_j | Y) from dataset for all attributes i != j
    mi_cond = {frozenset((column_i, column_j)):
               utils.conditional_mutual_information(
                   X[:, column_i], X[:, column_j], Y)
               for i, column_i in enumerate(range(num_col))
               for column_j in range(num_col)[i+1:]}

    # Randomly pick root
    root = np.random.choice(range(num_col))

    # Prim's algorithm
    vertices_in_tree = [False for _ in range(num_col)]
    vertices_in_tree[root] = True

    while sum(vertices_in_tree) < num_col:
        # Find minimum edge connecting a vertex not in the tree
        potential_edges = [frozenset((i, j))
         for i in range(num_col) if vertices_in_tree[i]
         for j in range(num_col) if not vertices_in_tree[j]]

        mis = [mi_cond[edge] for edge in potential_edges]
        idx = np.argmax(mis)
        best_edge = potential_edges[idx]
        # TODO: does frozenset preserve the order here?
        
        tuple_edge = tuple(best_edge)
        if vertices_in_tree[tuple_edge[0]]:
            vertices_in_tree[tuple_edge[1]] = True
            edges.append((tuple_edge[0], tuple_edge[1]))
        else:
            vertices_in_tree[tuple_edge[0]] = True
            edges.append((tuple_edge[1], tuple_edge[0]))
        
    return edges


# Sahami, M. (1996). Learning Limited Dependence Bayesian Classifiers. AAAI.
# https://dl.acm.org/doi/10.5555/3001460.3001537
def kdb(X, Y, k=1):
    """
    k: k
    Parents for each child sorted on cond. mutual information TODO: of what?
    """
    num_col = X.shape[1]
    # Calculate MI(X_i; Y) from dataset for all attributes
    miy = {column: utils.mutual_information(
        X[:, column], Y) for column in range(num_col)}

    # Calculate MI(X_i; X_j | Y) from dataset for all attributes i != j
    mi_cond = {frozenset((column_i, column_j)):
               utils.conditional_mutual_information(
                   X[:, column_i], X[:, column_j], Y)
               for i, column_i in enumerate(range(num_col))
               for column_j in range(num_col)[i+1:]}

    # Sort attributes on decreasing order of MI(X_i; Y)
    x_sorted = sorted(range(num_col), key=lambda x: miy[x], reverse=True)

    edges = []
    # For each Xi
    for i, xi in enumerate(x_sorted):
        # Add target Y as parent
        edges.append((num_col, xi))

        # Add min(i, k) parents with greatest mutual information conditioned
        # on the class, among those with greater mutual information with the
        # class
        m = [mi_cond[frozenset((xi, x_sorted[j]))]
            for j in range(i) if (xi, x_sorted[j]) not in edges]
        kth = min(i, k)
        list(map(lambda vk: edges.append((x_sorted[vk], xi)), np.argpartition(m, -kth)[-kth:]))

    return edges


def edges_to_parents(edges, p: int) -> List[List[int]]:
    """
    Parameters:
        edges: List of edges in a graph
        p: number of nodes in the graph

    Outputs:
        parents[i] contains parents of node i
    """
    parents: List[List[int]] = [[] for _ in range(p)]
    for parent, child in edges:
        parents[child].append(parent)
    return parents


def parents_to_edges(parents):
    """
    Parameters:
        parents: List of parents for each node in a graph

    Outputs:
        list of edges in the graph
    """
    p = len(parents)
    edges = []
    for child in range(p):
        for parent in parents[child]:
            edges.append((parent, child))
    return edges


# TODO: write own version?
def probs_to_pgmpy(node_probs, structure, p, cards):
    # Init pgmpy model
    pgmpy_model = BayesianNetwork(structure)
    cpds: list[TabularCPD | None] = [None for _ in range(p)]

    parents = [[] for _ in range(p)]
    for edge in structure:
        fro = edge[0]
        to = edge[1]
        parents[to].append(fro)

    for node in range(p):
        if node not in [u for edge in structure for u in edge]:
            pgmpy_model.add_node(node)
        cpt = node_probs[node].T
        pgmpy_cpd = TabularCPD(
            variable=node,
            variable_card=cards[node],
            evidence=parents[node],
            evidence_card=[cards[parent] for parent in parents[node]],
            values=cpt
        )

        #breakpoint()
        # Add cpds to model
        pgmpy_model.add_cpds(pgmpy_cpd)

    assert pgmpy_model.check_model()
    # Check that everything somes to 1 correctly

    return pgmpy_model


def pgmpy_infer(pgmpy_model, x_test, cards, variables=None):
    p = x_test.shape[1] + 1

    if variables is None:
        variables = range(p-1)
    # Exact inference
    pgmpy_infer = VariableElimination(pgmpy_model)

    pred = np.zeros((x_test.shape[0], cards[-1]))

    for i in range(x_test.shape[0]):
        point = x_test[i]
        evidence = {i: point[i] for i in variables}
        q = pgmpy_infer.query(variables=[p-1], evidence=evidence).values
        pred[i] = q

    return pred
