"""
Dataset Splitting and Random Seed Utilities.
Handles percentage-based train / validate / test splitting, seed generation,
and deterministic reproducible shuffling.
"""

import math
import random
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SplitConfig:
    train_pct: float
    valid_pct: float
    test_pct: float = 0.0
    seed: int = 42

    def validate(self) -> List[str]:
        """Validate percentage ranges and sum."""
        errors = []
        if self.train_pct <= 0:
            errors.append("Train percentage must be greater than 0%.")
        if self.valid_pct < 0:
            errors.append("Validation percentage cannot be negative.")
        if self.test_pct < 0:
            errors.append("Test percentage cannot be negative.")

        total = self.train_pct + self.valid_pct + self.test_pct
        if not math.isclose(total, 100.0, abs_tol=0.01):
            errors.append(f"Split percentages must sum to 100% (currently {total:.1f}%).")

        return errors


def generate_random_seed() -> int:
    """Generate a pseudo-random integer seed for reproducibility."""
    # 6-digit random number between 100,000 and 999,999
    return secrets.randbelow(900000) + 100000


def calculate_split_counts(total: int, train_pct: float, valid_pct: float, test_pct: float) -> Tuple[int, int, int]:
    """Calculate the exact number of rows allocated to each split."""
    if total <= 0:
        return 0, 0, 0

    if test_pct <= 0:
        n_train = int(round(total * (train_pct / 100.0)))
        n_valid = total - n_train
        n_test = 0
    else:
        n_train = int(round(total * (train_pct / 100.0)))
        n_valid = int(round(total * (valid_pct / 100.0)))
        n_test = total - n_train - n_valid
        # Edge case adjustments
        if n_test < 0:
            n_train += n_test
            n_test = 0

    # Ensure train has at least 1 record if total >= 1
    if n_train == 0 and total > 0:
        n_train = 1
        if n_valid > 0:
            n_valid -= 1
        elif n_test > 0:
            n_test -= 1

    return n_train, n_valid, n_test


def split_records(
    records: List[Dict[str, Any]],
    config: SplitConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Deterministically shuffle records with the given random seed and divide
    them into train, valid, and optional test splits.
    """
    errs = config.validate()
    if errs:
        raise ValueError("; ".join(errs))

    total = len(records)
    if total == 0:
        return {"train": [], "valid": []}

    # Deterministic shuffle using custom seed
    rng = random.Random(config.seed)
    indices = list(range(total))
    rng.shuffle(indices)

    n_train, n_valid, n_test = calculate_split_counts(
        total, config.train_pct, config.valid_pct, config.test_pct
    )

    train_indices = indices[:n_train]
    valid_indices = indices[n_train: n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    result = {
        "train": [records[i] for i in train_indices],
        "valid": [records[i] for i in valid_indices],
    }

    if config.test_pct > 0 and n_test > 0:
        result["test"] = [records[i] for i in test_indices]

    return result
