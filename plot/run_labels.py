#!/usr/bin/env python3
"""Build concise, collision-safe labels for compared run directories."""

from pathlib import Path
from typing import List, Sequence, Tuple


def _tokens(name: str) -> Tuple[str, ...]:
    return tuple(token for token in name.split("_") if token)


def _common_prefix_length(token_lists: Sequence[Tuple[str, ...]]) -> int:
    if not token_lists:
        return 0

    prefix_length = 0
    while all(
        prefix_length < len(tokens)
        and tokens[prefix_length] == token_lists[0][prefix_length]
        for tokens in token_lists
    ):
        prefix_length += 1
    return prefix_length


def _common_suffix_length(token_lists: Sequence[Tuple[str, ...]]) -> int:
    if not token_lists:
        return 0

    suffix_length = 0
    while all(
        suffix_length < len(tokens)
        and tokens[-1 - suffix_length] == token_lists[0][-1 - suffix_length]
        for tokens in token_lists
    ):
        suffix_length += 1
    return suffix_length


def _unique(values: Sequence[str]) -> bool:
    return len(set(values)) == len(values)


def _display_label(label: str) -> str:
    return label.replace("_", " ").title()


def _path_tail(path: Path, depth: int) -> str:
    return "/".join(path.parts[-depth:])


def _disambiguate(labels: Sequence[str], paths: Sequence[Path]) -> List[str]:
    """Add parent path components only where a shortened label collides."""
    result = list(labels)
    max_depth = max(len(path.parts) for path in paths)

    for depth in range(2, max_depth + 1):
        duplicate_indices = {
            index
            for index, label in enumerate(result)
            if result.count(label) > 1
        }
        if not duplicate_indices:
            return result
        for index in duplicate_indices:
            result[index] = _path_tail(paths[index], depth)
        if _unique(result):
            return result

    # This only occurs when the same directory is supplied more than once.
    # Keep labels readable while making each row/line distinguishable.
    seen = {}
    for index, label in enumerate(result):
        occurrence = seen.get(label, 0) + 1
        seen[label] = occurrence
        if occurrence > 1:
            result[index] = "{} #{}".format(label, occurrence)
    return result


def short_run_labels(run_directories: Sequence[Path]) -> List[str]:
    """Return labels formed from the parts that differ between run names.

    Names are split on underscores. The longest common token prefix and suffix
    are removed, so for example ``experiment_seed_100`` and
    ``experiment_seed_101`` become ``100`` and ``101``. If removing shared
    parts would make labels empty or ambiguous, the original name or enough
    parent path is retained to keep every label distinct.
    """
    paths = [
        Path(directory).expanduser().resolve()
        for directory in run_directories
    ]
    if not paths:
        return []

    names = [path.name for path in paths]
    if len(names) == 1:
        return [_display_label(names[0])]

    token_lists = [_tokens(name) for name in names]
    prefix_length = _common_prefix_length(token_lists)
    suffix_length = _common_suffix_length(token_lists)
    suffix_length = min(
        suffix_length,
        min(
            len(tokens) - prefix_length
            for tokens in token_lists
        ),
    )

    labels = []
    for name, tokens in zip(names, token_lists):
        end = len(tokens) - suffix_length if suffix_length else len(tokens)
        label = "_".join(tokens[prefix_length:end])
        labels.append(label or name)

    if _unique(labels):
        return [_display_label(label) for label in labels]
    return [
        _display_label(label)
        for label in _disambiguate(labels, paths)
    ]
