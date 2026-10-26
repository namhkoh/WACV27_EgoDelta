from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DELEAF_FACING_BY_LABEL = {
    "AD1": "right",
    "AD2": "left",
    "AD3": "right",
    "BD1": "left",
    "BD2": "right",
    "BD3": "right",
    "BD4": "right",
}


def parse_vector(value: str, *, length: int, name: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"{name} must have {length} comma-separated values")
    return np.asarray([float(part) for part in parts], dtype=float)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def feature_vector(
    *,
    goal_robot_m: np.ndarray,
    q_arm_rad: np.ndarray,
    dq_arm_rad_s: np.ndarray,
    tcp_position_robot_m: np.ndarray,
    phase_s: float | None = None,
    phase_fraction: float | None = None,
    horizon_s: float | None = None,
    horizon_fraction: float | None = None,
    deleaf_facing: str | None = None,
    include_phase_features: bool = False,
    include_gcsl_features: bool = False,
    include_deleaf_side_features: bool = False,
) -> np.ndarray:
    values = [
        goal_robot_m,
        goal_robot_m - tcp_position_robot_m,
        q_arm_rad,
        dq_arm_rad_s,
        tcp_position_robot_m,
    ]
    if include_phase_features:
        phase_value = 0.0 if phase_s is None else float(phase_s)
        fraction_value = 0.0 if phase_fraction is None else min(max(float(phase_fraction), 0.0), 1.0)
        values.append(np.asarray([phase_value, fraction_value, 1.0 - fraction_value], dtype=float))
    if include_gcsl_features:
        horizon_value = 0.0 if horizon_s is None else max(float(horizon_s), 0.0)
        fraction_value = 0.0 if horizon_fraction is None else min(max(float(horizon_fraction), 0.0), 1.0)
        values.append(np.asarray([horizon_value, fraction_value, 1.0 - fraction_value], dtype=float))
    if include_deleaf_side_features:
        facing = str(deleaf_facing or "").strip().lower()
        if facing not in ("left", "right"):
            raise ValueError("this policy requires --deleaf-facing or --deleaf-label to provide stem-facing side")
        values.append(
            np.asarray(
                [
                    1.0 if facing == "left" else 0.0,
                    1.0 if facing == "right" else 0.0,
                    -1.0 if facing == "left" else 1.0,
                ],
                dtype=float,
            )
        )
    return np.concatenate(values).reshape(1, -1)


def policy_uses_phase_features(metadata: dict[str, Any]) -> bool:
    names = metadata.get("input", {}).get("feature_names", [])
    return "phase_s" in names or "phase_fraction" in names


def policy_uses_gcsl_features(metadata: dict[str, Any]) -> bool:
    names = metadata.get("input", {}).get("feature_names", [])
    return "gcsl_horizon_s" in names or "gcsl_horizon_fraction" in names


def policy_uses_deleaf_side_features(metadata: dict[str, Any]) -> bool:
    names = metadata.get("input", {}).get("feature_names", [])
    return "deleaf_left_facing" in names or "deleaf_right_facing" in names


def resolve_deleaf_facing(label: str | None, facing: str | None) -> str | None:
    if label:
        key = str(label).strip().upper()
        if key not in DELEAF_FACING_BY_LABEL:
            raise ValueError(f"unknown deleaf label for side features: {label}")
        return DELEAF_FACING_BY_LABEL[key]
    if facing is None:
        return None
    value = str(facing).strip().lower()
    if value not in ("left", "right"):
        raise ValueError("--deleaf-facing must be left or right")
    return value


def predict_ridge(arrays: Any, x: np.ndarray) -> np.ndarray:
    x_norm = (x - arrays["ridge_x_mean"]) / arrays["ridge_x_std"]
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=float)], axis=1)
    y_norm = x_aug @ arrays["ridge_weights"]
    return y_norm * arrays["ridge_y_std"] + arrays["ridge_y_mean"]


def predict_bc(arrays: Any, metadata: dict[str, Any], x: np.ndarray) -> np.ndarray:
    training = metadata.get("training", {})
    layer_count = len(training.get("bc_hidden_dims", [])) + 1
    x_norm = (x - arrays["bc_x_mean"]) / arrays["bc_x_std"]
    activations = x_norm
    for index in range(layer_count):
        activations = activations @ arrays[f"bc_w{index}"] + arrays[f"bc_b{index}"]
        if index < layer_count - 1:
            activations = np.tanh(activations)
    return activations * arrays["bc_y_std"] + arrays["bc_y_mean"]


def predict_knn(arrays: Any, x: np.ndarray, *, k: int) -> tuple[np.ndarray, dict[str, Any]]:
    train_x = arrays["x"]
    train_y = arrays["y_delta"]
    train_norm = (train_x - arrays["knn_x_mean"]) / arrays["knn_x_std"]
    query_norm = (x - arrays["knn_x_mean"]) / arrays["knn_x_std"]
    distances = np.linalg.norm(train_norm - query_norm, axis=1)
    k = max(1, min(int(k), len(train_x)))
    nearest = np.argpartition(distances, k - 1)[:k]
    nearest = nearest[np.argsort(distances[nearest])]
    nearest_distances = distances[nearest]
    weights = 1.0 / np.maximum(nearest_distances, 1e-6)
    weights /= np.sum(weights)
    return np.sum(train_y[nearest] * weights.reshape(-1, 1), axis=0, keepdims=True), {
        "nearest_indices": nearest.astype(int).tolist(),
        "nearest_distances": nearest_distances.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
    }


def predict(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.policy_dir / "policy_metadata.json"
    arrays_path = args.policy_dir / "policy_arrays.npz"
    labels_path = args.policy_dir / "source_labels.json"
    metadata = read_json(metadata_path)
    labels = read_json(labels_path).get("source_labels", [])
    arrays = np.load(arrays_path, allow_pickle=False)

    q_arm = parse_vector(args.q_arm_rad, length=7, name="--q-arm-rad")
    dq_arm = (
        np.zeros(7, dtype=float)
        if args.dq_arm_rad_s is None
        else parse_vector(args.dq_arm_rad_s, length=7, name="--dq-arm-rad-s")
    )
    chunk_horizon_s = float(metadata["chunk_horizon_s"])
    phase_s = args.phase_s
    phase_fraction = args.phase_fraction
    if phase_fraction is None and phase_s is not None:
        phase_fraction = min(max(float(phase_s) / max(chunk_horizon_s, 1e-6), 0.0), 1.0)
    horizon_s = args.horizon_s
    horizon_fraction = args.horizon_fraction
    if policy_uses_gcsl_features(metadata):
        if horizon_s is None:
            phase_value = 0.0 if phase_s is None else float(phase_s)
            horizon_s = max(chunk_horizon_s - phase_value, 0.0)
        if horizon_fraction is None:
            horizon_fraction = min(max(float(horizon_s) / max(chunk_horizon_s, 1e-6), 0.0), 1.0)
    deleaf_facing = resolve_deleaf_facing(args.deleaf_label, args.deleaf_facing)
    if policy_uses_deleaf_side_features(metadata) and deleaf_facing is None:
        raise ValueError("policy was trained with deleaf side features; pass --deleaf-label or --deleaf-facing")

    x = feature_vector(
        goal_robot_m=parse_vector(args.goal_robot_m, length=3, name="--goal-robot-m"),
        q_arm_rad=q_arm,
        dq_arm_rad_s=dq_arm,
        tcp_position_robot_m=parse_vector(args.tcp_position_robot_m, length=3, name="--tcp-position-robot-m"),
        phase_s=phase_s,
        phase_fraction=phase_fraction,
        horizon_s=horizon_s,
        horizon_fraction=horizon_fraction,
        deleaf_facing=deleaf_facing,
        include_phase_features=policy_uses_phase_features(metadata),
        include_gcsl_features=policy_uses_gcsl_features(metadata),
        include_deleaf_side_features=policy_uses_deleaf_side_features(metadata),
    )
    neighbor_report: dict[str, Any] | None = None
    if args.mode == "bc":
        delta = predict_bc(arrays, metadata, x)
    elif args.mode == "ridge":
        delta = predict_ridge(arrays, x)
    else:
        knn_k = int(args.knn_k if args.knn_k is not None else metadata.get("training", {}).get("knn_k", 1))
        delta, neighbor_report = predict_knn(arrays, x, k=knn_k)

    chunk_size = int(metadata["chunk_size"])
    delta_chunk = delta.reshape(chunk_size, 7)
    q_chunk = q_arm.reshape(1, 7) + delta_chunk
    source_labels = None
    if neighbor_report is not None:
        source_labels = [labels[index] for index in neighbor_report["nearest_indices"] if index < len(labels)]
        neighbor_report["source_labels"] = source_labels
    return {
        "schema": "egodelta_action_chunk_prediction_v1",
        "policy_dir": str(args.policy_dir),
        "primitive": metadata.get("primitive"),
        "mode": args.mode,
        "sample_hz": float(metadata["sample_hz"]),
        "dt_s": float(1.0 / float(metadata["sample_hz"])),
        "chunk_size": chunk_size,
        "chunk_horizon_s": chunk_horizon_s,
        "goal_robot_m": parse_vector(args.goal_robot_m, length=3, name="--goal-robot-m").astype(float).tolist(),
        "q_arm_input_rad": q_arm.astype(float).tolist(),
        "phase_s": None if phase_s is None else float(phase_s),
        "phase_fraction": None if phase_fraction is None else float(phase_fraction),
        "gcsl_horizon_s": None if horizon_s is None else float(horizon_s),
        "gcsl_horizon_fraction": None if horizon_fraction is None else float(horizon_fraction),
        "deleaf_facing": deleaf_facing,
        "delta_q_arm_chunk_rad": delta_chunk.astype(float).tolist(),
        "q_arm_chunk_rad": q_chunk.astype(float).tolist(),
        "max_abs_first_step_delta_deg": float(math.degrees(np.max(np.abs(delta_chunk[0])))),
        "neighbor_report": neighbor_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit one action chunk from a trained EgoDelta policy artifact.")
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--goal-robot-m", required=True, help="Comma-separated x,y,z target in robot frame.")
    parser.add_argument("--q-arm-rad", required=True, help="Comma-separated current 7-DoF arm joint state.")
    parser.add_argument("--dq-arm-rad-s", default=None, help="Comma-separated current 7-DoF arm velocity; zeros if omitted.")
    parser.add_argument("--tcp-position-robot-m", required=True, help="Comma-separated current TCP x,y,z in robot frame.")
    parser.add_argument("--phase-s", type=float, default=None, help="Policy phase in seconds for phase-conditioned policies.")
    parser.add_argument("--phase-fraction", type=float, default=None, help="0..1 phase fraction for phase-conditioned policies.")
    parser.add_argument("--horizon-s", type=float, default=None, help="GCSL horizon-to-goal in seconds.")
    parser.add_argument("--horizon-fraction", type=float, default=None, help="0..1 GCSL remaining-horizon fraction.")
    parser.add_argument("--deleaf-label", default=None, help="Known deleaf target label such as BD1 or BD3; used for side-conditioned policies.")
    parser.add_argument("--deleaf-facing", choices=("left", "right"), default=None, help="Stem-facing side for side-conditioned deleaf policies.")
    parser.add_argument("--mode", choices=("bc", "knn", "ridge"), default="bc")
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = predict(args)
    if args.output_json is not None:
        write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
