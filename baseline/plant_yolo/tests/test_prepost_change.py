from yolo_aria_harvest_poc.prepost_change import combine_disappearance_scores


def test_single_recording_yolo_only_score_can_pass_threshold():
    score, weights = combine_disappearance_scores(
        yolo_absence_score=1.0,
        point_disappearance_score_value=0.0,
        point_disappearance_weight=0.0,
    )

    assert score == 1.0
    assert weights == {
        "yolo_absence_weight": 1.0,
        "point_disappearance_weight": 0.0,
    }


def test_default_balanced_score_preserves_previous_behavior():
    score, weights = combine_disappearance_scores(
        yolo_absence_score=1.0,
        point_disappearance_score_value=0.0,
        point_disappearance_weight=0.5,
    )

    assert score == 0.5
    assert weights == {
        "yolo_absence_weight": 0.5,
        "point_disappearance_weight": 0.5,
    }
