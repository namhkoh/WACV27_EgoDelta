from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from robot.policy.predict_action_chunk_policy import predict_bc, predict_knn, predict_ridge
from robot.kinematics.geometry import axis_angle_from_rotation, rotation_from_axis_angle


ARM_DOF = 7


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def predict_rows(
    *,
    arrays: Any,
    metadata: dict[str, Any],
    x: np.ndarray,
    mode: str,
    knn_k: int | None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if mode == "bc":
        return predict_bc(arrays, metadata, x), None
    if mode == "ridge":
        return predict_ridge(arrays, x), None
    if mode == "knn":
        k = int(knn_k if knn_k is not None else metadata.get("training", {}).get("knn_k", 1))
        return predict_knn(arrays, x, k=k)
    raise ValueError(f"unsupported mode: {mode}")


def training_arrays_required(arrays: Any, policy_dir: Path) -> None:
    required = {"x", "y_delta", "source_times_s"}
    missing = sorted(required.difference(arrays.files))
    if missing:
        raise RuntimeError(
            f"{policy_dir} does not contain exact training arrays ({', '.join(missing)} missing). "
            "Retrain that policy with --store-training-arrays, then rerun this evaluator. "
            "This is deliberate: exact training-data eval should not reconstruct a different dataset."
        )


def select_indices(
    labels: list[str],
    *,
    demo_label: str | None,
    row_index: int | None,
    max_rows: int | None,
) -> np.ndarray:
    if not labels:
        raise ValueError("source_labels.json is empty")
    selected_label = demo_label if demo_label is not None else labels[0]
    indices = np.asarray([index for index, label in enumerate(labels) if label == selected_label], dtype=int)
    if indices.size == 0:
        available = sorted(set(labels))
        raise ValueError(f"demo label {selected_label!r} not found. Available labels include: {available[:20]}")
    if row_index is not None:
        if row_index < 0 or row_index >= indices.size:
            raise ValueError(f"--row-index {row_index} is outside selected demo row count {indices.size}")
        indices = indices[row_index : row_index + 1]
    elif max_rows is not None:
        indices = indices[: max(1, int(max_rows))]
    return indices


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    policy_dir = args.policy_dir
    metadata = read_json(policy_dir / "policy_metadata.json")
    labels_payload = read_json(policy_dir / "source_labels.json")
    labels = [str(label) for label in labels_payload.get("source_labels", [])]
    arrays = np.load(policy_dir / "policy_arrays.npz", allow_pickle=False)
    training_arrays_required(arrays, policy_dir)

    chunk_size = int(metadata["chunk_size"])
    action_space = str(metadata.get("output", {}).get("action_space", "joint"))
    step_dim = 6 if action_space == "eef" else ARM_DOF
    action_dim = chunk_size * step_dim
    x = np.asarray(arrays["x"], dtype=float)
    y_delta = np.asarray(arrays["y_delta"], dtype=float)
    if y_delta.ndim != 2 or y_delta.shape[1] != action_dim:
        raise ValueError(
            f"y_delta shape {y_delta.shape} does not match chunk_size={chunk_size} action_space={action_space}"
        )
    if len(labels) != x.shape[0]:
        raise ValueError(f"source_labels length {len(labels)} does not match x rows {x.shape[0]}")

    selected = select_indices(
        labels,
        demo_label=args.demo_label,
        row_index=args.row_index,
        max_rows=args.max_rows,
    )
    pred, neighbor_report = predict_rows(
        arrays=arrays,
        metadata=metadata,
        x=x[selected],
        mode=args.mode,
        knn_k=args.knn_k,
    )
    target = y_delta[selected]
    pred = np.asarray(pred, dtype=float)
    n = int(len(selected))
    selected_labels = [labels[int(index)] for index in selected]
    source_times = np.asarray(arrays["source_times_s"], dtype=float)[selected]
    report_steps = min(int(args.report_steps), chunk_size)

    payload: dict[str, Any] = {
        "schema": "egodelta_policy_training_eval_v1",
        "policy_dir": str(policy_dir),
        "primitive": metadata.get("primitive"),
        "action_space": action_space,
        "mode": args.mode,
        "sample_hz": float(metadata["sample_hz"]),
        "chunk_size": chunk_size,
        "selected_label": selected_labels[0],
        "selected_row_count": n,
        "selected_source_time_s": source_times.astype(float).tolist(),
        "training_method": metadata.get("training", {}).get("training_method"),
        "feature_names": metadata.get("input", {}).get("feature_names", []),
        "neighbor_report": neighbor_report,
    }

    if action_space == "eef":
        pred_chunk = pred.reshape(n, chunk_size, 6)
        target_chunk = target.reshape(n, chunk_size, 6)
        pos_mm = (pred_chunk[:, :, :3] - target_chunk[:, :, :3]) * 1000.0
        rot_deg = np.zeros((n, chunk_size), dtype=float)
        for i in range(n):
            for s in range(chunk_size):
                r_pred = rotation_from_axis_angle(pred_chunk[i, s, 3:6])
                r_target = rotation_from_axis_angle(target_chunk[i, s, 3:6])
                rot_deg[i, s] = math.degrees(
                    float(np.linalg.norm(axis_angle_from_rotation(r_target.T @ r_pred)))
                )
        per_step_pos_mm = np.sqrt(np.mean(np.square(pos_mm), axis=(0, 2)))
        per_step_rot_deg = np.sqrt(np.mean(np.square(rot_deg), axis=0))
        payload.update(
            {
                "position_rmse_mm": {
                    "all_steps": float(np.sqrt(np.mean(np.square(pos_mm)))) if pos_mm.size else float("nan"),
                    "first_step": float(np.sqrt(np.mean(np.square(pos_mm[:, 0, :])))) if pos_mm.size else float("nan"),
                    "per_step": [float(v) for v in per_step_pos_mm[:report_steps]],
                    "max_abs_all_steps": float(np.max(np.abs(pos_mm))) if pos_mm.size else float("nan"),
                },
                "rotation_rmse_deg": {
                    "all_steps": float(np.sqrt(np.mean(np.square(rot_deg)))) if rot_deg.size else float("nan"),
                    "first_step": float(np.sqrt(np.mean(np.square(rot_deg[:, 0])))) if rot_deg.size else float("nan"),
                    "per_step": [float(v) for v in per_step_rot_deg[:report_steps]],
                    "max_abs_all_steps": float(np.max(np.abs(rot_deg))) if rot_deg.size else float("nan"),
                },
                "first_row": {
                    "source_label": selected_labels[0],
                    "source_time_s": float(source_times[0]),
                    "target_tcp_pose_first_step": target_chunk[0, 0].astype(float).tolist(),
                    "pred_tcp_pose_first_step": pred_chunk[0, 0].astype(float).tolist(),
                },
            }
        )
        return payload

    error = pred - target
    error_chunk = error.reshape(n, chunk_size, ARM_DOF)
    target_chunk = target.reshape(n, chunk_size, ARM_DOF)
    pred_chunk = pred.reshape(n, chunk_size, ARM_DOF)
    per_step_rmse_rad = np.sqrt(np.mean(np.square(error_chunk), axis=(0, 2)))
    per_step_target_rms_rad = np.sqrt(np.mean(np.square(target_chunk), axis=(0, 2)))
    first_step_error = error_chunk[:, 0, :]
    payload.update(
        {
            "rmse": {
                "all_steps_rad": rmse(error),
                "all_steps_deg": math.degrees(rmse(error)),
                "first_step_rad": rmse(first_step_error),
                "first_step_deg": math.degrees(rmse(first_step_error)),
                "first_step_per_joint_deg": [
                    math.degrees(float(value)) for value in np.sqrt(np.mean(np.square(first_step_error), axis=0))
                ],
                "per_step_deg": [math.degrees(float(value)) for value in per_step_rmse_rad[:report_steps]],
                "per_step_target_rms_deg": [
                    math.degrees(float(value)) for value in per_step_target_rms_rad[:report_steps]
                ],
            },
            "max_abs_error": {
                "all_steps_deg": math.degrees(float(np.max(np.abs(error)))) if error.size else float("nan"),
                "first_step_deg": math.degrees(float(np.max(np.abs(first_step_error)))) if first_step_error.size else float("nan"),
            },
            "first_row": {
                "source_label": selected_labels[0],
                "source_time_s": float(source_times[0]),
                "target_delta_q_arm_first_step_rad": target_chunk[0, 0].astype(float).tolist(),
                "pred_delta_q_arm_first_step_rad": pred_chunk[0, 0].astype(float).tolist(),
                "error_delta_q_arm_first_step_rad": error_chunk[0, 0].astype(float).tolist(),
            },
        }
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained action-chunk policy on exact saved training rows.")
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("bc", "ridge", "knn"), default="bc")
    parser.add_argument("--demo-label", default=None, help="Exact source label to evaluate. Defaults to the first label.")
    parser.add_argument("--row-index", type=int, default=None, help="Evaluate one row within the selected source label.")
    parser.add_argument("--max-rows", type=int, default=128, help="Max selected rows to evaluate unless --row-index is set.")
    parser.add_argument("--report-steps", type=int, default=20)
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = evaluate(args)
    if args.output_json is not None:
        write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
