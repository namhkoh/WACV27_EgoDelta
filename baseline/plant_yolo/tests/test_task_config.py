from yolo_aria_harvest_poc.task_config import load_task_config


def test_load_deleaf_config():
    deleaf = load_task_config("configs/deleaf.yaml")

    assert deleaf.task_name == "deleaf"
    assert deleaf.target_semantic_class == "petiole"
    assert deleaf.depth_band_m < 0.1
