from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_summarize_ablation_evaluates_primary_contact_proposals(tmp_path: Path) -> None:
    script = _load_script("scoring_ablation.py")
    outputs_root = tmp_path / "outputs"
    scene_dir = outputs_root / "full" / "egodelta_123456781234_tomato"
    scene_dir.mkdir(parents=True)
    (scene_dir / "contact_target_proposals.json").write_text(
        json.dumps(
            {
                "metadata": {"scene_name": scene_dir.name},
                "proposals": [
                    {
                        "proposal_id": 0,
                        "representative_point": [0.01, 0.0, 0.0],
                        "support_cluster_ids": [0],
                        "rank": 1,
                        "primary": True,
                    },
                    {
                        "proposal_id": 1,
                        "representative_point": [0.2, 0.0, 0.0],
                        "support_cluster_ids": [0],
                        "rank": 2,
                        "primary": False,
                    },
                ],
            }
        )
    )
    (scene_dir / "removed_clusters.json").write_text(
        json.dumps(
            [
                {
                    "cluster_id": 0,
                    "aabb_min": [0.0, 0.0, 0.0],
                    "aabb_max": [0.02, 0.02, 0.02],
                }
            ]
        )
    )
    gt = tmp_path / "gt.md"
    gt.write_text(
        "\n".join(
            [
                "1. 12345678-1234-1234-1234-123456789abc.vrs",
                "deleafing",
                "harvesting",
                "[0.0, 0.0, 0.0]",
                "",
            ]
        )
    )

    report = script.summarize_ablation(
        outputs_root=outputs_root,
        gt_path=gt,
        variants=["full"],
        threshold_m=0.03,
        selection_policy="primary",
    )

    tomato = report["variant_summaries"][0]["splits"]["Tomato"]
    assert tomato["Scenes"] == 1
    assert tomato["GT"] == 1
    assert tomato["Pred"] == 1
    assert tomato["TP"] == 1
    assert tomato["F1@3cm"] == 1.0
    organization = report["organization_summaries"][0]["splits"]["Tomato"]
    assert organization["target_bboxes"] == 1
    assert organization["selected_targets"] == 1


def test_ablation_runner_deep_merge_preserves_base_sections() -> None:
    script = _load_script("scoring_ablation.py")
    merged = script.deep_merge(
        {"ablation": {"scoring_mode": "full"}, "compute": {"device": "cuda"}},
        {"ablation": {"scoring_mode": "geometry_only"}},
    )

    assert merged["ablation"]["scoring_mode"] == "geometry_only"
    assert merged["compute"]["device"] == "cuda"


def test_reference_full_check_requires_reference_json() -> None:
    script = _load_script("scoring_ablation.py")
    report = {
        "metadata": {"threshold_m": 0.03, "selection_policy": "paper"},
        "variant_summaries": [{"variant": "full", "splits": {}}],
    }

    failures = script.check_reference_full(report)

    assert failures == ["reference full baseline check requires --reference-target-json"]


def test_reference_full_check_reports_metric_drift(tmp_path: Path) -> None:
    script = _load_script("scoring_ablation.py")
    gt, reference = _write_reference_target_fixture(tmp_path)
    splits = script.reference_full_baseline_from_target_json(reference, gt, 0.03)
    splits["All"]["TP"] -= 1
    report = {
        "metadata": {"threshold_m": 0.03, "selection_policy": "paper", "gt": str(gt)},
        "variant_summaries": [{"variant": "full", "splits": splits}],
    }

    failures = script.check_reference_full(report, reference)

    assert failures
    assert "All TP" in failures[0]


def test_reference_full_check_can_use_target_json(tmp_path: Path) -> None:
    script = _load_script("scoring_ablation.py")
    gt, reference = _write_reference_target_fixture(tmp_path)
    splits = script.reference_full_baseline_from_target_json(reference, gt, 0.03)
    report = {
        "metadata": {"threshold_m": 0.03, "selection_policy": "paper", "gt": str(gt)},
        "variant_summaries": [{"variant": "full", "splits": splits}],
    }

    assert script.check_reference_full(report, reference) == []


def _write_reference_target_fixture(tmp_path: Path) -> tuple[Path, Path]:
    gt = tmp_path / "gt.md"
    gt.write_text(
        "\n".join(
            [
                "1. 12345678-1234-1234-1234-123456789abc.vrs",
                "deleafing",
                "harvesting",
                "[0.0, 0.0, 0.0]",
                "",
            ]
        )
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "metadata": {},
                "targets": [
                    {
                        "scene": "egodelta_123456781234_tomato",
                        "target_x": 0.01,
                        "target_y": 0.0,
                        "target_z": 0.0,
                    }
                ],
            }
        )
    )
    return gt, reference


def test_run_eval_script_loads() -> None:
    script = _load_script("run_eval.py")
    assert script.E3_IDS["E3-3"] == "removed-side mirror"
    assert script.E3_IDS["E3-4"] == "binary delta in weights only"


def test_legacy_reproducibility_table_reports_published_and_rerun() -> None:
    script = _load_script("run_eval.py")
    table = script._legacy_reproducibility_table(
        {
            "published_f1": {"harvesting": 0.9074, "deleafing": 0.8824, "all": 0.8910},
            "rerun_f1": {
                "harvesting": 0.9074074074074074,
                "deleafing": 0.8780487804878049,
                "all": 0.8881789137380192,
            },
            "delta_f1": {
                "harvesting": 0.0000074074074074,
                "deleafing": -0.0043512195121951,
                "all": -0.0028210862619808,
            },
        }
    )

    assert "| deleafing | 0.882400 | 0.878049 | -0.004351 |" in table
    assert "| all | 0.891000 | 0.888179 | -0.002821 |" in table
