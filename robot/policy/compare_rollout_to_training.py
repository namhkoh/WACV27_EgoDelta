"""Compare a guarded BC rollout log against the policy's stored training rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARM_DOF = 7


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_policy_rows(policy_dir: Path, label: str | None) -> dict[str, Any]:
    metadata = read_json(policy_dir / "policy_metadata.json")
    labels_payload = read_json(policy_dir / "source_labels.json")
    arrays = np.load(policy_dir / "policy_arrays.npz", allow_pickle=True)
    feature_names = list(metadata.get("input", {}).get("feature_names", []))
    required = {"x", "y_delta", "source_times_s"}
    missing = sorted(required.difference(arrays.files))
    if missing:
        raise RuntimeError(f"policy arrays missing required fields: {missing}")
    q_cols = [feature_names.index(f"q_arm_rad.{index}") for index in range(ARM_DOF)]
    tcp_cols = [feature_names.index(f"tcp_position_robot_m.{axis}") for axis in "xyz"]
    source_labels = np.asarray([str(item) for item in labels_payload.get("source_labels", [])])
    if len(source_labels) != len(arrays["x"]):
        raise RuntimeError(
            f"source_labels length {len(source_labels)} does not match stored training rows {len(arrays['x'])}"
        )
    mask = np.ones(len(source_labels), dtype=bool)
    if label:
        mask = source_labels == label
    x = np.asarray(arrays["x"], dtype=float)[mask]
    y_delta = np.asarray(arrays["y_delta"], dtype=float)[mask].reshape(len(x), int(metadata["chunk_size"]), ARM_DOF)
    return {
        "metadata": metadata,
        "feature_names": feature_names,
        "times_s": np.asarray(arrays["source_times_s"], dtype=float)[mask],
        "labels": source_labels[mask],
        "q": x[:, q_cols],
        "tcp": x[:, tcp_cols],
        "y_delta": y_delta,
    }


def nearest_index(rows: np.ndarray, value: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(rows - value.reshape(1, -1), axis=1)))


def compare(policy_dir: Path, rollout_json: Path, label: str | None, endpoint_step: int) -> dict[str, Any]:
    policy = load_policy_rows(policy_dir, label=label)
    metadata = policy["metadata"]
    sample_hz = float(metadata["sample_hz"])
    rollout = read_json(rollout_json)
    iterations = rollout.get("rollout_iterations") or rollout.get("iterations", [])
    chunk_size = int(metadata["chunk_size"])
    step = max(0, min(int(endpoint_step), chunk_size - 1))
    rows: list[dict[str, Any]] = []
    for iteration in iterations:
        input_payload = iteration.get("input", {})
        prediction = iteration.get("prediction", {})
        q_arm = np.asarray(input_payload.get("q_arm_rad", []), dtype=float)
        tcp = np.asarray(input_payload.get("tcp_position_robot_m", []), dtype=float)
        q_chunk = np.asarray(prediction.get("q_arm_chunk_rad", []), dtype=float)
        if q_arm.shape != (ARM_DOF,) or tcp.shape != (3,) or q_chunk.ndim != 2 or len(q_chunk) == 0:
            continue
        step_i = min(step, len(q_chunk) - 1)
        nominal_index = int(round(float(input_payload.get("phase_s", 0.0)) * sample_hz))
        nominal_index = max(0, min(len(policy["q"]) - 1, nominal_index))
        nearest_q_index = nearest_index(policy["q"], q_arm)
        nearest_tcp_index = nearest_index(policy["tcp"], tcp)
        nominal_target_q = policy["q"][nominal_index] + policy["y_delta"][nominal_index, step_i]
        nearest_q_target_q = policy["q"][nearest_q_index] + policy["y_delta"][nearest_q_index, step_i]
        rows.append(
            {
                "iteration": int(iteration.get("iteration_index", len(rows) + 1)),
                "phase_s": float(input_payload.get("phase_s", 0.0)),
                "nominal_time_s": float(policy["times_s"][nominal_index]),
                "nearest_q_time_s": float(policy["times_s"][nearest_q_index]),
                "nearest_tcp_time_s": float(policy["times_s"][nearest_tcp_index]),
                "live_q_vs_nominal_deg": math.degrees(float(np.linalg.norm(q_arm - policy["q"][nominal_index]))),
                "live_q_vs_nearest_q_deg": math.degrees(float(np.linalg.norm(q_arm - policy["q"][nearest_q_index]))),
                "live_tcp_vs_nominal_m": float(np.linalg.norm(tcp - policy["tcp"][nominal_index])),
                "live_tcp_vs_nearest_tcp_m": float(np.linalg.norm(tcp - policy["tcp"][nearest_tcp_index])),
                "pred_endpoint_err_vs_nominal_deg": math.degrees(float(np.linalg.norm(q_chunk[step_i] - nominal_target_q))),
                "pred_endpoint_err_vs_nearest_q_deg": math.degrees(float(np.linalg.norm(q_chunk[step_i] - nearest_q_target_q))),
            }
        )
    sample_indices = []
    if rows:
        sample_indices = sorted(set([0, min(10, len(rows) - 1), min(20, len(rows) - 1), min(40, len(rows) - 1), len(rows) - 1]))
    return {
        "schema": "egodelta_rollout_training_compare_v1",
        "policy_dir": policy_dir,
        "rollout_json": rollout_json,
        "label_filter": label,
        "rollout_execution_error": rollout.get("execution_error"),
        "rollout_failed_guards": rollout.get("guards", {}).get("failed_guards"),
        "chunk_size": chunk_size,
        "endpoint_step": step,
        "iteration_count": len(rows),
        "median_pred_endpoint_err_vs_nominal_deg": float(np.median([row["pred_endpoint_err_vs_nominal_deg"] for row in rows])) if rows else None,
        "median_pred_endpoint_err_vs_nearest_q_deg": float(np.median([row["pred_endpoint_err_vs_nearest_q_deg"] for row in rows])) if rows else None,
        "sample_rows": [rows[index] for index in sample_indices],
        "final_row": rows[-1] if rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--rollout-json", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--endpoint-step", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(jsonable(compare(args.policy_dir, args.rollout_json, args.label, args.endpoint_step)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
