# SAM 3 Concept Masks + Aria MPS Geometry Baseline

This repository implements a from-scratch baseline for pre/post 3D change detection in tomato greenhouse manipulation tasks.

The baseline uses SAM 3 without fine-tuning. It uses text and optional exemplar concept prompts to segment task objects in already-extracted EgoDelta/Aria pinhole RGB frames, lifts those 2D masks into 3D using cached Aria/MPS world points, fuses multi-view evidence into 3D instances, compares pre/post scans in Aria world coordinates, and outputs task target points for harvesting and deleafing.

## Tasks

- `harvesting`: detects disappeared tomato fruit and outputs the tomato center plus a cut target. If reliable auxiliary peduncle/stem masks are present, the cut target uses that geometry; otherwise it is explicitly labeled as a tomato-top proxy, not a true peduncle cut point.
- `deleafing`: detects disappeared or changed tomato petiole / leaf-stem instances, not generic leaves, and outputs a stem-side cut target. If no main-stem auxiliary geometry is available, both ambiguous endpoints are reported.

## Install

Native SAM 3 backend:

```bash
bash scripts/setup_sam3_native.sh
```

Ultralytics SAM 3 backend:

```bash
bash scripts/setup_sam3_ultralytics.sh
```

`requirements.txt` intentionally does not pin torch. The setup scripts install torch separately so it can match the local CUDA environment. Native SAM 3 may require Hugging Face authentication and checkpoint access.

## Harvesting Example

```bash
python -m sam_aria_change_baseline.run_pipeline \
  --task harvesting \
  --mode extracted_frames \
  --pair-manifest /path/to/data/mps/egodelta_benchmark_data/pairs_fast/SCENE_ID/tomato_pair/manifest.json \
  --sam-backend native_sam3 \
  --frame-stride 1 \
  --aux-sam-score-thres 0.05 \
  --aux-min-mask-area-px 10 \
  --use-aux-prompts \
  --out outputs/harvest_scene_001
```

## Deleafing Example

```bash
python -m sam_aria_change_baseline.run_pipeline \
  --task deleafing \
  --mode extracted_frames \
  --pair-manifest /path/to/data/mps/egodelta_benchmark_data/pairs_fast/SCENE_ID/leaf_pair/manifest.json \
  --sam-backend native_sam3 \
  --frame-stride 1 \
  --deleafing-sam-score-thres 0.12 \
  --deleafing-min-mask-area-px 50 \
  --deleafing-max-mask-area-frac 0.03 \
  --deleafing-max-primary-masks-per-frame 12 \
  --deleafing-pipe-like-long-side-frac 0.35 \
  --deleafing-pipe-like-aspect-ratio 6.0 \
  --deleafing-primary-vertical-axis-thres 0.90 \
  --deleafing-primary-vertical-axis-image-direction image_horizontal \
  --petiole-min-points-inside-mask 4 \
  --petiole-dbscan-min-samples 3 \
  --petiole-instance-min-frames 1 \
  --petiole-min-3d-major-extent-m 0.04 \
  --petiole-max-3d-major-extent-m 0.55 \
  --out outputs/deleaf_scene_001
```

For deleafing, first check primary petiole masks without `--use-aux-prompts`. If those look good and you need stem-side endpoint selection, rerun with `--use-aux-prompts`. Raw SAM overlays are written to `visualizations/sam_pre_raw`; filtered petiole-like overlays are written to `visualizations/sam_pre`.

The deleafing path also applies a 3D major-axis extent gate after mask lifting and again after multi-view fusion. Width and height gates are not used. The 3D-passing masks are visualized in `visualizations/sam_pre_3d_petiole_filtered` and `visualizations/sam_post_3d_petiole_filtered`. Tune the extent gate with `--petiole-min-3d-major-extent-m` and `--petiole-max-3d-major-extent-m`.

The expected input is an already-extracted pre/post pair manifest such as `pairs_fast/<scene>/leaf_pair/manifest.json` or `pairs_fast/<scene>/tomato_pair/manifest.json`. The manifest points to rectified `frames_rect` images, `poses.csv`, `camera_pinhole.json`, and cached `points3D*.txt`. The pipeline writes only `frames/pre_frames.json` and `frames/post_frames.json` metadata under the output directory; it does not re-extract frames from VRS. Deleafing primary masks also have a principal-axis filter that rejects thin main-stem-like components before 3D lifting. As a final cleanup, disconnected components inside each remaining primary mask are checked independently, and compact leaf-like or main-stem-like components are removed before the mask is saved. For EgoDelta-rectified frames, real vertical structure often appears image-horizontal, so use `--deleafing-primary-vertical-axis-image-direction image_horizontal` when applying that filter. Principal-axis overlays are written to `visualizations/sam_pre_principal_axes` and `visualizations/sam_pre_all_principal_axes`.

Current deleafing primary prompts are side-stem/petiole-specific and intentionally avoid generic `leaf`:

```text
side stem of tomato plant, which is not a main stem and hung horizontally
```

Auxiliary deleafing prompts are only for selecting the stem-side endpoint and are intentionally worded to avoid leaves and branches:

```text
main stem of tomato plant, without any leaves or branches
```

## Deleafing Cut-Point Evaluation

Evaluate predicted `final_predictions.json` files against benchmark 3D cut-point labels with one-to-one matching at 2 cm:

```bash
python -m sam_aria_change_baseline.evaluate_cut_points \
  --task deleafing \
  --pred-root outputs/deleaf_eval_runs \
  --gt-labels /path/to/data/mps/egodelta_benchmark_splitted/3d_points_labels.txt \
  --out outputs/deleaf_cutpoint_eval
```

The evaluator loads `target_point_3d_world` from each prediction, matches predictions to GT cut points with a one-to-one distance threshold, and writes:

- `per_scene_metrics.json`
- `per_scene_metrics.csv`
- `aggregate_metrics.json`
- `cutpoint_eval_results.json`

For a single prediction file, pass its scene id explicitly:

```bash
python -m sam_aria_change_baseline.evaluate_cut_points \
  --task deleafing \
  --pred-root outputs/deleaf/final_predictions.json \
  --scene-id 040bc88b-3760-4426-b512-1ac710c45eac \
  --out outputs/deleaf_cutpoint_eval_single
```

## Exemplar Config

Exemplars are optional. If a backend adapter does not support them, the pipeline warns and continues with text-only prompts.

```yaml
harvesting:
  positive_exemplar_crops:
    - path/to/tomato_crop_01.jpg
  positive_exemplar_boxes:
    - image_path: path/to/frame.jpg
      box_xyxy: [100, 200, 180, 280]
```

## Outputs

The main outputs are:

- `final_predictions.json`
- `debug_stats.json`
- `sam_predictions/*_predictions.json`
- `sam_predictions/*_predictions_filtered.json`
- `lifted_3d/*_lifted_masks.json`
- `lifted_3d/*_instances.json`
- `change_candidates.json`
- `visualizations/sam_pre/*.jpg`
- `visualizations/sam_post/*.jpg`
- `visualizations/lifting_debug/pre/*.jpg`
- `visualizations/lifting_debug/post/*.jpg`
- `visualizations/target_overlay/pre/*.jpg`
- `visualizations/target_overlay/post/*.jpg`
- `visualizations/pre_instances.ply`
- `visualizations/post_instances.ply`
- `visualizations/disappeared_instances.ply`
- `visualizations/target_points.ply`
- `visualizations/all_debug_scene.ply`
- `visualizations/change_debug_scene.ply`
- `visualizations/change_centroids_and_links.ply`
- `visualizations/disappeared_vs_post_scene.ply`
- `visualizations/change_debug_summary.json`
- `visualizations/change_debug_pre/*.png`
- `visualizations/change_debug_post/*.png`
- `visualizations/summary.png`

The change-debug PLY files are for inspecting 3D pre/post differences. `change_debug_scene.ply` colors pre instances by change status, overlays post instances in green, and adds target/centroid markers. `change_centroids_and_links.ply` contains only centroid spheres and sampled links to matched or nearest post instances, which is usually easier to inspect. `disappeared_vs_post_scene.ply` shows disappeared pre-instance points against post-instance points.

The change-debug PNG folders project those same 3D decisions back into RGB frames. `change_debug_pre` shows pre instances colored by change status; `change_debug_post` overlays post instances in green and projects disappeared pre instances in yellow where they should appear if still present.

## Limitations

- SAM 3 may hallucinate or segment wrong objects from text prompts.
- Thin structures like petioles, peduncles, and leaf stems may be difficult.
- Aria semi-dense points may be sparse on shiny tomatoes and thin stems.
- Harvesting cut points are only peduncle-aware when auxiliary stem/peduncle masks are reliable.
- Without reliable peduncle/stem segmentation, harvesting targets are geometry proxies from the tomato top surface.
- Thin auxiliary structures often need lower thresholds, for example `--aux-sam-score-thres 0.05 --aux-min-mask-area-px 10`.
- Deleafing uses more sensitive defaults than harvesting because petioles are thin and sparsely supported. If petioles are still missed, try `--deleafing-sam-score-thres 0.05 --deleafing-min-mask-area-px 10 --petiole-min-points-inside-mask 2`.
- Deleafing auxiliary main-stem prompts are intentionally off in `scripts/run_deleaf_scene.sh` unless `USE_AUX_PROMPTS=1`. The pipeline uses vertical central-stem wording and rejects auxiliary masks/instances that look like short lateral petioles or peduncles.
- Pre/post comparison requires shared world coordinates.
- This baseline does not train or fine-tune SAM.
