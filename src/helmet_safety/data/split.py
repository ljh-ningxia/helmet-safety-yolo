"""Deterministic, class-stratified subset sampling.

Why stratification: a naive ``random.sample`` on an imbalanced detection set
can easily drop most of the rare ``no-hardhat`` images, which are precisely the
positive alarm cases. We bucket images by (box-count bin, contains-violation)
and sample proportionally inside each bucket with a fixed seed.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, TypeVar

T = TypeVar("T")


def _box_bin(n: int) -> str:
    if n == 0:
        return "b0"
    if n == 1:
        return "b1"
    if n <= 3:
        return "b2_3"
    if n <= 8:
        return "b4_8"
    return "b9plus"


def stratified_indices(items: Sequence[T], group_key, seed: int,
                       n: int | None = None) -> List[int]:
    """Return indices of a representative subset.

    ``group_key(index, item)`` returns a hashable bucket label. If ``n`` is
    None or >= len(items), all indices are returned unchanged.
    """
    if n is None or n >= len(items):
        return list(range(len(items)))
    if n <= 0:
        return []

    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, item in enumerate(items):
        buckets[group_key(i, item)].append(i)

    rng = random.Random(seed)
    for key in buckets:
        rng.shuffle(buckets[key])

    # Largest-remainder allocation keeps the exact target size without bias.
    total = len(items)
    exact = {k: n * len(v) / total for k, v in buckets.items()}
    take = {k: min(int(exact[k]), len(v)) for k, v in buckets.items()}
    remainder = n - sum(take.values())
    order = sorted(buckets, key=lambda k: exact[k] - int(exact[k]), reverse=True)
    for k in order:
        if remainder <= 0:
            break
        if take[k] < len(buckets[k]):
            take[k] += 1
            remainder -= 1
    # Any residual (e.g. tiny buckets) is filled round-robin.
    k_iter = 0
    while remainder > 0 and order:
        k = order[k_iter % len(order)]
        if take[k] < len(buckets[k]):
            take[k] += 1
            remainder -= 1
        k_iter += 1
        if k_iter > 10000:
            break

    chosen: List[int] = []
    for k, v in buckets.items():
        chosen.extend(v[:take[k]])
    chosen.sort()
    return chosen


def describe_strata(items: Sequence[T], indices: Sequence[int], group_key) -> Dict[str, Dict[str, int]]:
    full = Counter(group_key(i, items[i]) for i in range(len(items)))
    sub = Counter(group_key(i, items[i]) for i in indices)
    return {k: {"full": full[k], "subset": sub.get(k, 0)} for k in sorted(full)}
