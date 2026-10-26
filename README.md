# EgoDelta — WACV 2027 code

EgoDelta recovers 3D robot action targets from egocentric pre/post scene changes.
This package comes from `redacted/WACV27-camera-ready` at
`1659525f888125cfa8bef706c57da1dc0cbe329f`.

| Component | Entry point |
| --- | --- |
| Selected WACV target-recovery method | [target_extraction/README.md](target_extraction/README.md) |
| Evaluation and reproduction | [target_extraction/REPRODUCTION.md](target_extraction/REPRODUCTION.md) |
| Gaussian splatting | [train.py](train.py), [render.py](render.py), [gaussian_renderer/](gaussian_renderer/) |
| RBY1 kinematics and execution | [robot/kinematics/README.md](robot/kinematics/README.md) |
| Robot policy training | [train_action_chunk_policy.py](robot/policy/train_action_chunk_policy.py) |
| Offline policy evaluation | [eval_policy_on_training_data.py](robot/policy/eval_policy_on_training_data.py) |
| Physical-trial evaluation | [robot/evaluation/README.md](robot/evaluation/README.md) |
| New repository setup | [REPOSITORY_IMPORT.md](REPOSITORY_IMPORT.md) |

The pinned CUDA extension sources, including GLM, are bundled as ordinary
directories under `submodules/`. Tracked reference and evaluation policy
checkpoints are included as real files.

For the reconstruction environment, use `bash scripts/setup_env.sh` with Conda
and the CUDA/PyTorch toolchain in [environment.yml](environment.yml).
Robot workflows require separately installed SDK and camera dependencies.

Raw recordings, the recorded policy-action dataset, Stage-2 caches, and the
RB-Y1 SDK are not included. The selected Stage-3 evaluation requires external
frozen Stage-2 caches and ground truth. Inherited filesystem paths and calibration
examples must be adapted to the destination machine. The packaging scope is
documented in [REPOSITORY_IMPORT.md](REPOSITORY_IMPORT.md).

The original branch overview is preserved in [README_SOURCE.md](README_SOURCE.md).
Existing licenses and attribution are retained; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This archive establishes the source release. It does not certify a fresh GPU
training run or physical-robot evaluation.

Public-release configuration: `/path/to/...`, `C:/path/to/...`, and `/home/user/...`
are example paths, and `192.0.2.x` addresses are documentation-only placeholders.
Supply your own paths and network endpoints through the existing command-line
options. Camera recording and Pi05 rollout require `--base-serial` and
`--wrist-serial`, or the `BASE_CAMERA_SERIAL` and `LEFT_WRIST_CAMERA_SERIAL`
environment variables. Source branch namespaces marked `redacted/` are anonymized.
