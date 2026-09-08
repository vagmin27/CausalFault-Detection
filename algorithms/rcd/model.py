# Localized Constraint-Based Causal Discovery (RCD).
# Based on Ikram et al. (2022):
# "Root Cause Analysis of Failures in Microservices through Causal Discovery"
#
# Implements:
# - F-node (failure/intervention indicator: 0=normal, 1=anomalous).
# - Localized conditional independence testing (Fisher-Z / partial correlation).
# - Local neighborhood restriction around the failure indicator.
# - Top-k root-cause ranking.

import math
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from scipy import stats


def partial_correlation(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    # Computes sample partial correlation corr(x, y | Z).
    # If Z is empty, returns Pearson correlation.
    if Z is None or Z.size == 0 or Z.shape[1] == 0:
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        r, _ = stats.pearsonr(x, y)
        return float(r) if not np.isnan(r) else 0.0

    # Linear regression residuals: e_x = x - Z w_x, e_y = y - Z w_y
    try:
        # Add intercept to Z
        Z_design = np.column_stack([np.ones(len(Z)), Z])
        w_x, _, _, _ = np.linalg.lstsq(Z_design, x, rcond=None)
        res_x = x - Z_design @ w_x

        w_y, _, _, _ = np.linalg.lstsq(Z_design, y, rcond=None)
        res_y = y - Z_design @ w_y

        if np.std(res_x) < 1e-8 or np.std(res_y) < 1e-8:
            return 0.0

        r, _ = stats.pearsonr(res_x, res_y)
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


def fisher_z_test(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    alpha: float = 0.05
) -> Tuple[bool, float, float]:
    # Fisher's Z-transform conditional independence test:
    # H0: corr(x, y | Z) = 0.
    # Returns (is_independent, p_value, z_statistic).
    n = len(x)
    k = Z.shape[1] if (Z is not None and Z.size > 0) else 0
    dof = n - k - 3

    if dof <= 0:
        return False, 0.0, 1.0

    r = partial_correlation(x, y, Z)
    r = max(-0.9999, min(0.9999, r))

    z_stat = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(dof)
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    is_indep = (p_val > alpha)
    return is_indep, float(p_val), float(abs(z_stat))
