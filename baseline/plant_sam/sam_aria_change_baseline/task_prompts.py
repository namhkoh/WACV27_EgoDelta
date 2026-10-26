from __future__ import annotations

import os


TASK_PROMPTS = {
    "harvesting": {
        "primary": [
            "tomato fruit",
            "ripe tomato",
            "green tomato",
            "tomato",
        ],
        "auxiliary": [
            "tomato stem",
            "tomato peduncle",
            "tomato pedicel",
            "fruit pedicel",
            "fruit stem",
            "stem connected to tomato",
            "tomato calyx",
            "calyx",
            "tomato truss",
            "tomato cluster stem",
        ],
        "target_class": "tomato",
    },
    "deleafing": {
        "primary": [
            # "tomato petiole",
            # "petiole",
            # "tomato leaf petiole",
            # "leaf petiole",
            # "leaf stalk",
            # "tomato branch from main stem",
            "side stem of tomato plant, which is not a main stem and hung horizontally",
            # "side branch of tomato plant",
        ],
        "auxiliary": [
            "main stem of tomato plant, without any leaves or branches",
            # "vertical main stem of tomato plant",
            # "main vertical tomato stem",
        ],
        "target_class": "petiole_or_leaf_stem",
    },
}


def _env_prompt_override(task: str, group: str) -> list[str] | None:
    env_name = f"SAM_ARIA_{task.upper()}_{group.upper()}_PROMPTS"
    raw = os.environ.get(env_name)
    if not raw:
        return None
    prompts = [item.strip() for item in raw.split("||") if item.strip()]
    return prompts or None


def prompts_for_task(task: str, use_auxiliary: bool = False) -> list[tuple[str, str]]:
    """Return (prompt, semantic_group) pairs for a task."""
    if task not in TASK_PROMPTS:
        raise ValueError(f"Unsupported task {task!r}.")
    primary = _env_prompt_override(task, "primary") or TASK_PROMPTS[task]["primary"]
    pairs = [(prompt, "primary") for prompt in primary]
    if use_auxiliary:
        auxiliary = _env_prompt_override(task, "auxiliary") or TASK_PROMPTS[task]["auxiliary"]
        pairs.extend((prompt, "auxiliary") for prompt in auxiliary)
    return pairs


def target_class_for_group(task: str, semantic_group: str) -> str:
    if semantic_group == "primary":
        return TASK_PROMPTS[task]["target_class"]
    return "auxiliary_structure"
