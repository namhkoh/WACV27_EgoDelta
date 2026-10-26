from pathlib import Path

from egodelta_diff.data import find_scene_targets, parse_gt_markdown


def test_gt_parser_and_scene_lookup(tmp_path: Path):
    path = tmp_path / "gt.md"
    path.write_text("1. abcdef00-0000.vrs\n\ndeleafing\n[1, 2, 3]\n\nharvesting\n[4, 5, 6]\n")
    entries = parse_gt_markdown(path)
    found = find_scene_targets("egodelta_abcdef00_leaf", entries)
    assert found is not None
    assert found[1] == "deleafing"
    assert found[2].tolist() == [[1.0, 2.0, 3.0]]
