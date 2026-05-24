from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def latest_time_dir(case_dir: str | Path) -> Path:
    """Return the highest numeric OpenFOAM time directory."""
    root = Path(case_dir)
    candidates: list[tuple[float, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            candidates.append((float(path.name), path))
        except ValueError:
            continue
    if not candidates:
        raise ValueError(f"no numeric time directories found in {root}")
    return max(candidates, key=lambda item: item[0])[1]


def read_internal_vector_field(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    lines = _nonuniform_lines(text, field_name="internalField", value_type="vector")
    values = np.empty((len(lines), 3), dtype=float)
    for idx, line in enumerate(lines):
        values[idx] = np.fromstring(line.strip().strip("()"), sep=" ")
    return values


def read_internal_scalar_field(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    return np.asarray(
        [float(line.strip()) for line in _nonuniform_lines(text, field_name="internalField", value_type="scalar")],
        dtype=float,
    )


def read_patch_scalar_values(
    path: str | Path,
    patch: str,
    *,
    expected_count: int | None = None,
) -> np.ndarray:
    """Read a scalar patch value list from an OpenFOAM surfaceScalarField."""
    text = Path(path).read_text(encoding="utf-8")
    body = "\n".join(_patch_body_lines(text, patch))
    nonuniform_match = re.search(r"value\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(", body)
    if nonuniform_match:
        n_values = int(nonuniform_match.group(1))
        tail = body[nonuniform_match.end() :]
        values: list[float] = []
        for line in tail.splitlines():
            stripped = line.strip()
            if not stripped or stripped in {")", ");", ";"}:
                continue
            values.append(float(stripped))
            if len(values) == n_values:
                break
        if len(values) != n_values:
            raise ValueError(f"expected {n_values} values for patch {patch}, found {len(values)}")
        return np.asarray(values, dtype=float)

    uniform_match = re.search(r"value\s+uniform\s+([-+0-9.eE]+)\s*;", body)
    if uniform_match:
        if expected_count is None:
            raise ValueError(f"patch {patch} is uniform; pass expected_count to expand it")
        return np.full(expected_count, float(uniform_match.group(1)), dtype=float)

    raise ValueError(f"could not find scalar value for patch {patch} in {path}")


def _nonuniform_lines(text: str, *, field_name: str, value_type: str) -> list[str]:
    match = re.search(rf"{field_name}\s+nonuniform\s+List<{value_type}>\s+(\d+)\s*\(", text)
    if not match:
        raise ValueError(f"could not find nonuniform List<{value_type}> for {field_name}")
    n_values = int(match.group(1))
    values: list[str] = []
    for line in text[match.end() :].splitlines():
        stripped = line.strip()
        if not stripped or stripped in {")", ");", ";"}:
            continue
        values.append(stripped)
        if len(values) == n_values:
            break
    if len(values) != n_values:
        raise ValueError(f"expected {n_values} {value_type} values, found {len(values)}")
    return values


def _patch_body_lines(text: str, patch: str) -> list[str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != patch:
            continue
        brace_idx = _next_nonempty_line(lines, idx + 1)
        if brace_idx is None or lines[brace_idx].strip() != "{":
            continue
        depth = 1
        body: list[str] = []
        for body_line in lines[brace_idx + 1 :]:
            depth += body_line.count("{")
            depth -= body_line.count("}")
            if depth == 0:
                return body
            body.append(body_line)
    raise ValueError(f"patch {patch} not found")


def _next_nonempty_line(lines: list[str], start: int) -> int | None:
    for idx in range(start, len(lines)):
        if lines[idx].strip():
            return idx
    return None
