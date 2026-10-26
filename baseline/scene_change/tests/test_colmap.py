from pathlib import Path
import struct

from egoscene.colmap import (
    ColmapImage,
    read_cameras_binary,
    read_images_binary,
    read_images_txt,
    read_points3d_binary,
    write_images_txt,
)


def test_colmap_image_round_trip(tmp_path: Path):
    rows = [
        ColmapImage(
            9,
            1.0,
            0.0,
            0.0,
            0.0,
            0.1,
            0.2,
            0.3,
            1,
            "frame.jpg",
            ((12.5, 24.0, 101), (3.0, 4.0, -1)),
        )
    ]
    path = tmp_path / "images.txt"
    write_images_txt(path, rows)
    loaded = read_images_txt(path)
    assert len(loaded) == 1
    assert loaded[0].image_id == 1
    assert loaded[0].name == "frame.jpg"
    assert loaded[0].tz == 0.3
    assert loaded[0].points2d == rows[0].points2d


def test_colmap_binary_readers(tmp_path: Path):
    camera_path = tmp_path / "cameras.bin"
    camera_path.write_bytes(
        struct.pack("<QiiQQdddd", 1, 7, 1, 640, 480, 500.0, 501.0, 320.0, 240.0)
    )
    cameras = read_cameras_binary(camera_path)
    assert cameras[0].camera_id == 7
    assert cameras[0].model == "PINHOLE"
    assert cameras[0].params == (500.0, 501.0, 320.0, 240.0)

    image_path = tmp_path / "images.bin"
    image_header = struct.pack("<idddddddi", 3, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 7)
    image_path.write_bytes(
        struct.pack("<Q", 1)
        + image_header
        + b"frame.jpg\x00"
        + struct.pack("<Qddq", 1, 12.5, 24.0, 101)
    )
    images = read_images_binary(image_path)
    assert images[0].name == "frame.jpg"
    assert images[0].points2d == ((12.5, 24.0, 101),)

    points_path = tmp_path / "points3D.bin"
    points_path.write_bytes(
        struct.pack("<Q", 1)
        + struct.pack("<QdddBBBdQii", 101, 1.0, 2.0, 3.0, 10, 20, 30, 0.25, 1, 3, 0)
    )
    points = read_points3d_binary(points_path)
    assert points[0].point_id == 101
    assert points[0].rgb == (10, 20, 30)
    assert points[0].track == ((3, 0),)
