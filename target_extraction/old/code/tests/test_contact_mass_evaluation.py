from egodelta_diff.evaluation import classify_failure, finalize_failure_labels


def _row(**override):
    return {
        "scene_id": "scene", "task": "deleafing", "matched": True,
        "m_gt_ret": 1.0, "m_gt_rho": 1.0, "d_gt": 0.01,
        "n_gt_in_c": 1, "n_c_per_gt": 1, "ext_norm": 1.0,
        "conf_c": 1.0, "zero_mass_cause": None,
    } | override


def _class(row, ret, rho):
    return classify_failure(row, ret_norm=ret, rho_norm=rho,
                            ret_cutoff=0.1, rho_cutoff=0.1, extent_cutoff=3)[0]


def test_failure_order_and_a1_a2_are_separate():
    assert _class(_row(d_gt=0.07), 0.01, 0.01) == "D"
    assert _class(_row(n_gt_in_c=2), 0.01, 0.01) == "C"
    assert _class(_row(), 0.01, 1.0) == "A1"
    assert _class(_row(), 1.0, 0.01) == "A2"


def test_normalizer_guard_and_deterministic_rationale():
    rows = [_row(), _row(scene_id="failure", matched=False, m_gt_ret=0.01)]
    finalize_failure_labels(rows)
    assert rows[1]["normalization_low_confidence"] is True
    assert rows[1]["failure_class"] == "A1"
    assert "m_gt_ret_norm=" in rows[1]["rationale"]
    assert rows[1]["human_label"] == ""
