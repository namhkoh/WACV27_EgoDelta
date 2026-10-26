# Aria MPS Pre/Post Change Detection PoC

This repository implements proof-of-concept baselines for localizing disappeared greenhouse targets in 3D from Project Aria pre/post recordings.

The Aria harvesting pipeline itself uses YOLO only for inference. It does not fine-tune on Project Aria data and does not require a custom tomato dataset. It loads a tomato YOLO11 segmentation checkpoint, lifts tomato masks into 3D with Aria MPS semi-dense points, compares pre/post scans, and outputs disappeared tomato regions, 3D centroids, and a weak proxy harvesting cut point.

The same Aria geometry pipeline can also run zero-shot/promptable baselines for:

- `harvest`: tomato disappearance and tomato-top proxy cut point.
- `deleaf`: petiole disappearance and petiole-base proxy cut point.

The upstream tomato repository documents a `best.pt` release URL, but as of May 15, 2026 the GitHub repository has no published releases. If the release download returns 404, train a replacement checkpoint from the public Laboro Tomato dataset before running the Aria pipeline.

The cut point is explicitly labeled as `proxy_cut_point_from_tomato_top_surface`. It is not a true peduncle or stem cut point because no peduncle/keypoint model is used.

## Install

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

For YOLO training on the local RTX 3090/CUDA 12.x machine, use the pinned training repair file if `torch`/`torchvision` imports fail:

```bash
bash scripts/fix_torch_training_env.sh
```

Try to download the upstream checkpoint:

```bash
bash scripts/download_tomato_model.sh
```

The upstream README documents this URL:

```bash
wget -O models/best.pt "https://github.com/Fonyuy45/yolo11-tomato-segmentation/releases/download/v1.0/best.pt"
```

If it returns 404, prepare `models/best.pt` by training on the public Laboro Tomato dataset.

## Prepare Tomato YOLO Checkpoint

Download the Laboro Tomato dataset from Roboflow Universe. This requires a Roboflow API key because the dataset is hosted through Roboflow.

```bash
export ROBOFLOW_API_KEY=...
python scripts/download_laboro_tomato_dataset.py
```

By default this writes:

```text
datasets/Laboro_Tomato/
data/laboro_tomato.yaml
```

Train YOLO11 segmentation and copy the best checkpoint to `models/best.pt`:

```bash
bash scripts/train_tomato_yolo.sh
```

If training fails during `from ultralytics import YOLO` with an error from `torchvision` or `torch._dynamo`, reinstall the pinned compatible PyTorch/TorchVision pair:

```bash
bash scripts/fix_torch_training_env.sh
```

Equivalent explicit command:

```bash
python -m yolo_aria_harvest_poc.train_yolo_tomato \
  --data data/laboro_tomato.yaml \
  --model x \
  --epochs 100 \
  --batch 4 \
  --imgsz 640 \
  --project training_runs \
  --name tomato_yolo11_seg \
  --copy-to models/best.pt
```

Use a smaller model or fewer epochs for a quick smoke test:

```bash
YOLO_MODEL_SIZE=n EPOCHS=3 BATCH=4 bash scripts/train_tomato_yolo.sh
```

Set `DEVICE=0` to force a GPU, or omit `DEVICE` to let the training wrapper choose CUDA when available and CPU otherwise.

## Run: Single Recording

Use this mode when one VRS/MPS recording contains both pre-harvest and post-harvest time ranges.

```bash
python -m yolo_aria_harvest_poc.run_pipeline \
  --task harvest \
  --mode single_recording \
  --vrs /path/to/scene.vrs \
  --mps /path/to/mps \
  --pre-start-s 0 \
  --pre-end-s 22 \
  --post-start-s 33 \
  --post-end-s 52 \
  --model models/best.pt \
  --rgb-stream-id 214-1 \
  --frame-stride 10 \
  --out outputs/scene_001 \
  --world-up 0,0,1
```

To match the EgoDelta rectified Aria view, add:

```bash
  --image-view egodelta_pinhole \
  --aria-camera-label camera-rgb \
  --rect-size 720 \
  --rect-focal 380.0
```

This saves the same clockwise-rotated 720x720 pinhole RGB frames used by EgoDelta and projects MPS points with the matching rotated pinhole calibration. Do not mix these rectified frames with raw Aria RGB projection.

The original proxy cut-point heuristic remains the default:

```bash
--cutpoint-mode proxy
```

To use a separate peduncle model, pass one of the peduncle cut-point modes:

```bash
python -m yolo_aria_harvest_poc.run_pipeline \
  --task harvest \
  --mode single_recording \
  --vrs /path/to/scene.vrs \
  --mps /path/to/mps \
  --pre-start-s 0 \
  --pre-end-s 22 \
  --post-start-s 33 \
  --post-end-s 52 \
  --model models/best.pt \
  --peduncle-model models/peduncle.pt \
  --cutpoint-mode peduncle_auto \
  --out outputs/scene_001
```

Cut-point modes:

- `proxy`: current tomato-top-surface heuristic.
- `peduncle_keypoint`: requires peduncle keypoints and predicts the cut point from the fruit attachment toward the stem-side keypoint.
- `peduncle_mask`: requires peduncle masks, lifts the mask to 3D, fits a PCA curve, and chooses the tomato-facing endpoint plus an offset along the peduncle axis.
- `peduncle_auto`: tries keypoints, then masks, then falls back to `proxy`.

Peduncle keypoint names can be set in model output order:

```bash
--peduncle-keypoint-names fruit_attachment,peduncle_mid,stem_attachment,cut_point
```

## Download Peduncle Dataset

The Wageningen/4TU dataset for `3D pose estimation of tomato peduncle nodes using deep keypoint detection and point cloud` contains RGB images, depth images, and pose annotations for tomato peduncle nodes.

```bash
bash scripts/download_wur_peduncle_dataset.sh
```

By default this writes:

```text
datasets/wur_peduncle_keypoints/
```

Set `UNZIP_DATASET=0` if you only want the zip:

```bash
UNZIP_DATASET=0 bash scripts/download_wur_peduncle_dataset.sh
```

## Train Peduncle YOLO

The peduncle trainer supports both pose/keypoint and segmentation models. For the WUR/4TU keypoint dataset, first convert the COCO-style JSON annotations to Ultralytics YOLO pose format:

```bash
python scripts/convert_wur_peduncle_to_yolo_pose.py \
  --src datasets/wur_peduncle_keypoints/keypoints_dataset \
  --out datasets/peduncle_pose_yolo \
  --yaml-out data/peduncle_pose.yaml
```

This creates:

```text
datasets/peduncle_pose_yolo/images/train
datasets/peduncle_pose_yolo/images/val
datasets/peduncle_pose_yolo/labels/train
datasets/peduncle_pose_yolo/labels/val
data/peduncle_pose.yaml
```

The WUR keypoint order is `Mtop, JP, Mbut, Pu`.

Train a keypoint model:

```bash
TASK=pose DATA_YAML=data/peduncle_pose.yaml DEVICE=1 bash scripts/train_peduncle_yolo.sh
```

This writes:

```text
models/peduncle_pose.pt
```

Train a segmentation model:

```bash
cp data/peduncle_segment.example.yaml data/peduncle_segment.yaml
# edit data/peduncle_segment.yaml so path points to your converted dataset
TASK=segment DATA_YAML=data/peduncle_segment.yaml DEVICE=1 bash scripts/train_peduncle_yolo.sh
```

This writes:

```text
models/peduncle_segment.pt
```

Explicit command:

```bash
python -m yolo_aria_harvest_poc.train_yolo_peduncle \
  --task pose \
  --data data/peduncle_pose.yaml \
  --model-size s \
  --epochs 200 \
  --batch 8 \
  --imgsz 1024 \
  --device 1 \
  --project training_runs \
  --name peduncle_pose_yolo11s \
  --copy-to models/peduncle_pose.pt
```

Use the trained model in the harvesting pipeline:

```bash
--peduncle-model models/peduncle_pose.pt \
--cutpoint-mode peduncle_auto \
--peduncle-keypoint-names Mtop,JP,Mbut,Pu
```

## Run: Separate Recordings

Use this mode only when the pre/post MPS results are already in the same world coordinate frame, for example from Aria Multi-SLAM.

```bash
python -m yolo_aria_harvest_poc.run_pipeline \
  --task harvest \
  --mode separate_recordings \
  --pre-vrs /path/to/pre.vrs \
  --post-vrs /path/to/post.vrs \
  --pre-mps /path/to/pre_mps \
  --post-mps /path/to/post_mps \
  --model models/best.pt \
  --rgb-stream-id 214-1 \
  --frame-stride 10 \
  --out outputs/scene_001 \
  --world-up 0,0,1
```

If comparable MPS world-frame metadata is present and incompatible, the pipeline fails with:

```text
Pre and post MPS are not in the same world frame. Run Aria Multi-SLAM or provide an alignment transform.
```

This first version does not implement ICP alignment.

## Extending to Deleafing

Harvesting currently uses a tomato YOLO segmentation checkpoint. Deleafing can run a YOLO-only zero-shot baseline with YOLOE segmentation and the same Aria MPS pre/post 3D pipeline.

Supported tasks:

```bash
python -m yolo_aria_harvest_poc.run_pipeline --task harvest ...
python -m yolo_aria_harvest_poc.run_pipeline --task deleaf ...
```

Deleafing example:

```bash
python -m yolo_aria_harvest_poc.run_pipeline \
  --task deleaf \
  --task-config configs/deleaf.yaml \
  --mode single_recording \
  --vrs /path/to/scene.vrs \
  --mps /path/to/mps \
  --pre-start-s 0 \
  --pre-end-s 22 \
  --post-start-s 33 \
  --post-end-s 52 \
  --segmentation-backend yoloe_prompt \
  --yoloe-model yoloe-11s-seg.pt \
  --frame-stride 10 \
  --imgsz 1024 \
  --world-up 0,0,1 \
  --out outputs/scene_001_deleaf \
  --debug
```

YOLOE prompts are configured in `configs/deleaf.yaml`. The default deleaf prompts target petioles and leaf stalks, not individual leaf blades:

```text
petiole
leaf petiole
tomato petiole
tomato leaf petiole
petiole of tomato leaf
base of leaf petiole
leaf stem
tomato leaf stem
leaf stalk
tomato leaf stalk
thin leaf stem
thin green leaf stem
small green stem connecting leaf to main stem
```

Deleafing output is written to `final_predictions_deleaf.json`. It contains disappeared petiole instances and `proxy_cut_point_from_petiole_base`. This is a proxy endpoint from 3D petiole geometry, not a trained petiole keypoint prediction.

Task configs live in:

```text
configs/harvest.yaml
configs/deleaf.yaml
```

## Outputs

The pipeline writes:

```text
outputs/scene_001/
  frames/
  yolo_predictions/
  segmentation_predictions/
  lifted_3d/
  visualizations/
  disappeared_tomatoes.json
  disappeared_petioles.json
  proxy_cut_points.json
  final_predictions.json
  final_predictions_deleaf.json
  debug_stats.json
```

Key files:

- `final_predictions.json`: removed tomatoes, disappeared tomato 3D centroids, proxy cut points, scores, and confidence.
- `debug_stats.json`: frame counts, detection counts, lifted 3D counts, instance counts, warnings, and point cloud stats.
- `visualizations/*.ply`: pre tomato points, post tomato points, disappeared tomato points, proxy cut points, and a combined debug scene.
- `visualizations/summary.png`: compact run summary.
- `visualizations/deleaf/*.ply`: pre/post petiole instances, disappeared petioles, cut points, and optional main-stem axis.

Inspect PLY files in MeshLab, CloudCompare, or Open3D.

## Method

1. Extract RGB frames from Project Aria VRS.
2. Run YOLO11 tomato segmentation on pre/post frames.
3. Collapse tomato ripeness classes into semantic class `tomato`.
4. Project Aria MPS semi-dense world points into the same camera view used for inference, either raw Aria RGB or EgoDelta-compatible rectified pinhole RGB.
5. Keep points inside each tomato mask.
6. Apply a front-surface depth band to reduce background contamination.
7. Cluster per-detection 3D points with DBSCAN.
8. Fuse per-frame 3D detections into tomato instances.
9. Compare pre/post instances with visibility gating.
10. Estimate cut points using the configured mode. The proxy mode estimates `proxy_cut_point_from_tomato_top_surface`; peduncle modes estimate from associated lifted peduncle keypoints or mask curves.

## Limitations

- A checkpoint trained on Laboro Tomato may not generalize to Aria egocentric greenhouse imagery.
- Aria semi-dense points can be sparse on smooth or shiny tomato surfaces.
- True harvesting cut-point detection requires peduncle, stem, or keypoint detection.
- The proxy cut point is a geometry heuristic near the top/attachment side of a disappeared tomato.
- This PoC is mainly for removed tomato localization and weak proxy cut-point estimation.
- Deleafing promptable segmentation may confuse petioles with stems, strings, leaf veins, or leaf blades.
- True deleafing cut-point detection requires petiole keypoint labels or a trained petiole model.

## Tests

```bash
pytest -q
```
