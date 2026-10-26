from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: tuple[tuple[float, float, int], ...] = ()

    def line(self, *, image_id: int | None = None, name: str | None = None) -> str:
        values = (
            image_id if image_id is not None else self.image_id,
            self.qw,
            self.qx,
            self.qy,
            self.qz,
            self.tx,
            self.ty,
            self.tz,
            self.camera_id,
            name if name is not None else self.name,
        )
        return " ".join(str(x) for x in values)


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapPoint3D:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: tuple[tuple[int, int], ...]


_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_exact(handle, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise EOFError(f"Expected {count} bytes, got {len(data)}")
    return data


def read_cameras_binary(path: str | Path) -> list[ColmapCamera]:
    cameras = []
    with Path(path).open("rb") as handle:
        count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", _read_exact(handle, 24))
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            model, param_count = _CAMERA_MODELS[model_id]
            params = struct.unpack("<" + "d" * param_count, _read_exact(handle, 8 * param_count))
            cameras.append(ColmapCamera(camera_id, model, width, height, tuple(params)))
    return cameras


def read_images_binary(path: str | Path) -> list[ColmapImage]:
    images = []
    with Path(path).open("rb") as handle:
        count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        for _ in range(count):
            values = struct.unpack("<idddddddi", _read_exact(handle, 64))
            name_bytes = bytearray()
            while True:
                value = _read_exact(handle, 1)
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            point_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
            points2d = tuple(struct.unpack("<ddq", _read_exact(handle, 24)) for _ in range(point_count))
            images.append(
                ColmapImage(
                    image_id=values[0],
                    qw=values[1],
                    qx=values[2],
                    qy=values[3],
                    qz=values[4],
                    tx=values[5],
                    ty=values[6],
                    tz=values[7],
                    camera_id=values[8],
                    name=name_bytes.decode("utf-8"),
                    points2d=points2d,
                )
            )
    return images


def read_points3d_binary(path: str | Path) -> list[ColmapPoint3D]:
    points = []
    with Path(path).open("rb") as handle:
        count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        for _ in range(count):
            values = struct.unpack("<QdddBBBdQ", _read_exact(handle, 51))
            track = tuple(struct.unpack("<ii", _read_exact(handle, 8)) for _ in range(values[8]))
            points.append(
                ColmapPoint3D(
                    point_id=values[0],
                    xyz=(values[1], values[2], values[3]),
                    rgb=(values[4], values[5], values[6]),
                    error=values[7],
                    track=track,
                )
            )
    return points


def write_cameras_txt(path: str | Path, cameras: list[ColmapCamera]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for camera in cameras:
        fields = (camera.camera_id, camera.model, camera.width, camera.height, *camera.params)
        lines.append(" ".join(str(x) for x in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_points3d_txt(
    path: str | Path,
    points: list[ColmapPoint3D],
    *,
    image_ids: set[int] | None = None,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    written = 0
    for point in points:
        track = point.track if image_ids is None else tuple(pair for pair in point.track if pair[0] in image_ids)
        if image_ids is not None and not track:
            continue
        fields = (point.point_id, *point.xyz, *point.rgb, point.error, *(x for pair in track for x in pair))
        lines.append(" ".join(str(x) for x in fields))
        written += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def read_images_txt(path: str | Path) -> list[ColmapImage]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records: list[ColmapImage] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed COLMAP image row in {path}: {line}")
        # COLMAP stores one POINTS2D row after every image row. It may be empty.
        points2d: tuple[tuple[float, float, int], ...] = ()
        if i < len(lines):
            point_fields = lines[i].split()
            i += 1
            if len(point_fields) % 3:
                raise ValueError(f"Malformed COLMAP POINTS2D row in {path}: {lines[i - 1]}")
            points2d = tuple(
                (float(point_fields[j]), float(point_fields[j + 1]), int(point_fields[j + 2]))
                for j in range(0, len(point_fields), 3)
            )
        records.append(
            ColmapImage(
                image_id=int(fields[0]),
                qw=float(fields[1]),
                qx=float(fields[2]),
                qy=float(fields[3]),
                qz=float(fields[4]),
                tx=float(fields[5]),
                ty=float(fields[6]),
                tz=float(fields[7]),
                camera_id=int(fields[8]),
                name=fields[9],
                points2d=points2d,
            )
        )
    return records


def write_images_txt(path: str | Path, records: list[ColmapImage], *, reindex: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for image_id, record in enumerate(records, start=1):
        point_row = " ".join(str(x) for point in record.points2d for x in point)
        lines.extend((record.line(image_id=image_id if reindex else record.image_id), point_row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
