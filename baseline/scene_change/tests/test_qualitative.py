from __future__ import annotations

import pytest

from egoscene.qualitative import select_stratified_tasks


def _tasks() -> list[dict[str, str]]:
    return [
        {
            "task_id": f"{kind}_{scene}_{view}",
            "scene_id": f"{kind}_{scene}",
            "task_kind": kind,
        }
        for kind in ("deleafing", "harvesting")
        for scene in range(4)
        for view in range(3)
    ]


def test_stratified_selection_is_deterministic_and_scene_distinct() -> None:
    first = select_stratified_tasks(_tasks(), scenes_per_kind=3, seed=17)
    second = select_stratified_tasks(_tasks(), scenes_per_kind=3, seed=17)
    assert first == second
    for rows in first.values():
        assert len(rows) == 3
        assert len({row["scene_id"] for row in rows}) == 3


def test_stratified_selection_rejects_insufficient_scenes() -> None:
    with pytest.raises(ValueError, match="Need 5 deleafing scenes"):
        select_stratified_tasks(_tasks(), scenes_per_kind=5)
