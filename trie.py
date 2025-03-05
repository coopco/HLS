import numpy as np
import numpy.typing as npt
from typing import Optional, List


class TrieNode:
    def __init__(self, cardinality: int, target_card: int):
        self.children: List[Optional[TrieNode]] = [None for _ in range(cardinality)]
        self.cardinality = cardinality
        #self.counts: npt.NDArray = np.zeros(cardinality)
        self.counts: npt.NDArray = np.zeros(target_card)

# TODO: have an underlying np.matrix for efficiency
class Trie:
    def __init__(self, cardinalities: npt.NDArray):
        self.target_card = cardinalities[-1]
        self.root = TrieNode(cardinalities[0], self.target_card)
        self.cardinalities = cardinalities

    def insert(self,
               sequence: npt.NDArray,
               count: int =1) -> None :
        node = self.root
        for i, num in enumerate(sequence[:-1]):
            node.counts[sequence[-1]] += count
            if node.children[num] is None:
                node.children[num] = TrieNode(self.cardinalities[i+1], self.target_card)
            node = node.children[num]
        node.counts[sequence[-1]] += count

    def insert_set(self,
                   dataset: npt.NDArray,
                   idxs: Optional[List[int]] = None):
        """
            Assumes target_idx is idxs[-1]
        """
        if idxs is None:
            #idxs = np.linspace(0, dataset.shape[1], 1)
            idxs = list(range(dataset.shape[1]))

        for row in dataset: self.insert(row[idxs])

    # TODO: return int if full sequence given
    def search(self,
               sequence) -> npt.NDArray:
        node = self.root
        for num in sequence:
            if node.children[num] is None:
                return np.zeros(self.target_card)  # Sequence not found
            node = node.children[num]
        return node.counts

    def print_leaves(self):
        self._print_leaves(self.root, [])

    def _print_leaves(self, node: Optional[TrieNode] = None, current_path = []) -> None:
        if node is None:
            return

        print("Path:", current_path, "Counts:", node.counts)

        for num, child_node in enumerate(node.children):
            self._print_leaves(child_node, current_path + [num])


