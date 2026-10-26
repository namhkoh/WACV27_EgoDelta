from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass(frozen=True)
class PlyHeader:
    encoding: str
    vertex_count: int
    vertex_properties: tuple[tuple[str, str], ...]
    data_offset: int


def read_vertex_fields(path: str | Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    """Read selected scalar vertex fields without requiring ``plyfile``.

    EgoDelta's Gaussian PLYs are binary little-endian files with a scalar-only
    vertex element. ASCII files are accepted as a convenience for tests and
    small hand-authored fixtures.
    """

    path = Path(path)
    requested = tuple(dict.fromkeys(fields))
    with path.open("rb") as handle:
        header = _read_header(handle, path)
        names = {name for name, _ in header.vertex_properties}
        missing = sorted(set(requested) - names)
        if missing:
            raise ValueError(f"{path}: missing vertex fields: {', '.join(missing)}")

        if header.encoding == "ascii":
            return _read_ascii_vertices(handle, header, requested, path)
        return _read_binary_vertices(handle, header, requested, path)


def read_gaussian_projection_fields(path: str | Path) -> dict[str, np.ndarray]:
    """Load centers plus optional Gaussian footprint/opacity fields."""

    path = Path(path)
    with path.open("rb") as handle:
        header = _read_header(handle, path)
    names = {name for name, _ in header.vertex_properties}
    required = ["x", "y", "z"]
    optional = [name for name in ("scale_0", "scale_1", "opacity") if name in names]
    data = read_vertex_fields(path, required + optional)
    count = len(data["x"])

    if "scale_0" in data and "scale_1" in data:
        raw_scale = np.maximum(data["scale_0"], data["scale_1"]).astype(np.float64)
        data["world_scale"] = np.exp(np.clip(raw_scale, -20.0, 5.0)).astype(np.float32)
    else:
        data["world_scale"] = np.full(count, 0.003, dtype=np.float32)

    if "opacity" in data:
        raw_opacity = np.asarray(data["opacity"], dtype=np.float64)
        data["alpha"] = (1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -30.0, 30.0)))).astype(
            np.float32
        )
    else:
        data["alpha"] = np.ones(count, dtype=np.float32)
    return data


def _read_header(handle: BinaryIO, path: Path) -> PlyHeader:
    first = handle.readline()
    if first.strip() != b"ply":
        raise ValueError(f"{path}: not a PLY file")

    encoding: str | None = None
    current_element: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []

    while True:
        raw = handle.readline()
        if not raw:
            raise ValueError(f"{path}: PLY header has no end_header")
        try:
            line = raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: malformed PLY header") from exc
        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue
        parts = line.split()
        if parts[0] == "format":
            if len(parts) < 2:
                raise ValueError(f"{path}: malformed format line")
            encoding = parts[1]
        elif parts[0] == "element":
            if len(parts) != 3:
                raise ValueError(f"{path}: malformed element line")
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError(f"{path}: list-valued vertex properties are unsupported")
            if parts[1] not in _PLY_DTYPES:
                raise ValueError(f"{path}: unsupported PLY type {parts[1]!r}")
            vertex_properties.append((parts[2], parts[1]))
        elif parts[0] == "end_header":
            break

    if encoding not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"{path}: unsupported PLY encoding {encoding!r}")
    if vertex_count is None or not vertex_properties:
        raise ValueError(f"{path}: missing vertex element or properties")
    return PlyHeader(
        encoding=encoding,
        vertex_count=vertex_count,
        vertex_properties=tuple(vertex_properties),
        data_offset=handle.tell(),
    )


def _read_binary_vertices(
    handle: BinaryIO,
    header: PlyHeader,
    requested: tuple[str, ...],
    path: Path,
) -> dict[str, np.ndarray]:
    endian = "<" if header.encoding == "binary_little_endian" else ">"
    dtype = np.dtype(
        [(name, endian + _PLY_DTYPES[type_name]) for name, type_name in header.vertex_properties]
    )
    vertices = np.fromfile(handle, dtype=dtype, count=header.vertex_count)
    if len(vertices) != header.vertex_count:
        raise ValueError(
            f"{path}: expected {header.vertex_count} vertices, found {len(vertices)}"
        )
    return {name: np.asarray(vertices[name], dtype=np.float32).copy() for name in requested}


def _read_ascii_vertices(
    handle: BinaryIO,
    header: PlyHeader,
    requested: tuple[str, ...],
    path: Path,
) -> dict[str, np.ndarray]:
    property_names = [name for name, _ in header.vertex_properties]
    requested_indices = {name: property_names.index(name) for name in requested}
    output = {name: np.empty(header.vertex_count, dtype=np.float32) for name in requested}
    for row_index in range(header.vertex_count):
        raw = handle.readline()
        if not raw:
            raise ValueError(f"{path}: truncated ASCII vertex data at row {row_index}")
        values = raw.split()
        if len(values) != len(property_names):
            raise ValueError(f"{path}: malformed ASCII vertex row {row_index}")
        for name, column in requested_indices.items():
            output[name][row_index] = float(values[column])
    return output
