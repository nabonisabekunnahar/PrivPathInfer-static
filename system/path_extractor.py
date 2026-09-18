"""
path_extractor.py — flatten a trained decision tree into root-to-leaf paths.

Each path is an ordered list of conditions the record must satisfy to
reach its leaf, plus the leaf's predicted label. No encryption or
privacy logic belongs here — this is a plain traversal of the sklearn
tree structure.
"""

from dataclasses import dataclass, field
from typing import List

from sklearn.tree import _tree


@dataclass(frozen=True)
class Condition:
    feature_index: int
    threshold: float
    direction: str  # "left" (feature <= threshold) or "right" (feature > threshold)


@dataclass
class Path:
    path_id: int
    conditions: List[Condition]
    label: int
    depth: int


def extract_paths(clf) -> List[Path]:
    """
    Flatten a fitted sklearn.tree.DecisionTreeClassifier into its
    root-to-leaf paths, visiting every internal node exactly once.

    Args:
        clf: fitted DecisionTreeClassifier

    Returns:
        List[Path], in the tree's natural left-to-right (declaration)
        order.
    """
    tree = clf.tree_
    paths: List[Path] = []

    def recurse(node_id, conditions):
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            label = int(clf.classes_[tree.value[node_id].argmax()])
            paths.append(
                Path(
                    path_id=len(paths),
                    conditions=list(conditions),
                    label=label,
                    depth=len(conditions),
                )
            )
            return

        feature_index = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])

        recurse(
            tree.children_left[node_id],
            conditions + [Condition(feature_index, threshold, "left")],
        )
        recurse(
            tree.children_right[node_id],
            conditions + [Condition(feature_index, threshold, "right")],
        )

    recurse(0, [])
    return paths
