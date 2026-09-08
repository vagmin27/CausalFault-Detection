# Isolation Tree (iTree) and Bilateral Weighting Model for BWOAIF.
# Based on Hannák et al. (2023):
# "Bilateral-Weighted Online Adaptive Isolation Forest for anomaly detection in streaming data"

import math
import numpy as np
from typing import Optional, List, Dict, Any


def c_factor(n: int) -> float:
    # Average path length of unsuccessful search in BST for sample size n.
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.0
    euler_mascheroni = 0.5772156649
    return 2.0 * (math.log(n - 1) + euler_mascheroni) - (2.0 * (n - 1) / n)


class ITreeNode:
    # Node in an Isolation Tree.

    def __init__(
        self,
        left=None,
        right=None,
        split_feature: Optional[int] = None,
        split_value: Optional[float] = None,
        size: int = 0,
        is_leaf: bool = False,
    ):
        self.left: Optional[ITreeNode] = left
        self.right: Optional[ITreeNode] = right
        self.split_feature: Optional[int] = split_feature
        self.split_value: Optional[float] = split_value
        self.size: int = size
        self.is_leaf: bool = is_leaf


class StreamingIsolationTree:
    # Individual Isolation Tree with age tracking and isolation path computation.

    def __init__(self, max_height: int = 8, sub_sample_size: int = 64):
        self.max_height = max_height
        self.sub_sample_size = sub_sample_size
        self.root: Optional[ITreeNode] = None
        self.age_batch: int = 0
        self.anomaly_performance_weight: float = 1.0

    def fit(self, X: np.ndarray, current_batch: int = 0) -> None:
        # Constructs an isolation tree on a sample of observations.
        self.age_batch = current_batch
        n_samples = len(X)
        if n_samples == 0:
            return

        if n_samples > self.sub_sample_size:
            idx = np.random.choice(n_samples, self.sub_sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        self.root = self._build_tree(X_sample, current_height=0)

    def _build_tree(self, X: np.ndarray, current_height: int) -> ITreeNode:
        n_samples, n_features = X.shape
        if current_height >= self.max_height or n_samples <= 1:
            return ITreeNode(size=n_samples, is_leaf=True)

        # Select random feature
        feat = np.random.randint(0, n_features)
        min_val, max_val = float(np.min(X[:, feat])), float(np.max(X[:, feat]))

        if min_val >= max_val:
            return ITreeNode(size=n_samples, is_leaf=True)

        split = float(np.random.uniform(min_val, max_val))
        left_mask = X[:, feat] < split
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return ITreeNode(size=n_samples, is_leaf=True)

        left_child = self._build_tree(X[left_mask], current_height + 1)
        right_child = self._build_tree(X[right_mask], current_height + 1)

        return ITreeNode(
            left=left_child,
            right=right_child,
            split_feature=feat,
            split_value=split,
            size=n_samples,
            is_leaf=False,
        )

    def path_length(self, x: np.ndarray) -> float:
        # Returns path length h(x) to isolate observation x.
        if self.root is None:
            return 0.0

        current = self.root
        length = 0.0
        while current is not None:
            if current.is_leaf or current.split_feature is None:
                return length + c_factor(current.size)

            length += 1.0
            val = x[current.split_feature]
            if val < current.split_value:
                current = current.left
            else:
                current = current.right

        return length
