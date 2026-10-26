# EgoMask: EgoDelta image-mask correction

EgoMask turns the exported 3D oracle components into editable 2D instance-mask
initializations, selects 12 useful pre-action views per scene–intervention unit,
and divides the 720 resulting images evenly among four annotators. It is a small
Python HTTP server plus a browser canvas; the browser needs no installation.

The dataset has **30 unique scene UUIDs**. Each has a deleafing unit and a
harvesting unit, giving 60 scene–intervention units and 720 masks
(30 scenes × 2 interventions × 12 images). The run produces exactly **180
images per annotator**. Each image is assigned to one person, with three images
from every unit allocated to each annotator.

## What the labels mean

Each task shows one selected pre-action RGB image and a colored projection of
the scene's exported 3D components. The final PNG is an 8-bit instance label
map:

- `0`: background;
- `1`: oracle component 0;
- `2`: oracle component 1;
- and so on.

The export at
`oracle_component_masks_gt_adjacent_dbscan_e478073` is a **GT-adjacent
diagnostic oracle**, not a deployable prediction. Human-corrected image masks
derived from it must be described as annotation/evaluation ground truth, never
as an output available to EGODELTA at deployment time.

## One-time setup for the coordinator

From the repository root:

```bash
cd annotation_tool
conda create -n egomask python=3.10 -y
conda activate egomask
python -m pip install -r requirements.txt
```

Prepare the full run. Replace the four IDs with short, stable names if desired:

```bash
python -m egomask.prepare \
  --output /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --images-per-scene 12 \
  --annotators annotator_1 annotator_2 annotator_3 annotator_4 \
  --workers 4
```

Preparation validates all scene/image/camera mappings, projects the 3D
components with a conservative pre-model depth test, and chooses mask-visible
views across the capture timeline. It does not alter the source dataset. A
partially interrupted preparation can be continued with the same command plus
`--resume`; a completed run is never overwritten.

Start the shared server:

```bash
python -m egomask.serve \
  --run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --final-export /path/to/data/EgoDelta/annotation_runs/image_masks_gt_wacv2027_human_polygon_candidate_20260817 \
  --host 0.0.0.0 \
  --port 8765
```

When `--final-export` is provided, every valid private token also gets a
read-only **View all 720 final masks** button. This global browser is not
limited to the token owner's 180 editing tasks. It organizes all 720 masks by
the 60 scene–intervention units and shows, side by side, the RGB frame, an
opacity-adjustable final-mask overlay, and the standalone binary mask. It never
writes to either the annotation run or the frozen export.
The button is the normal entry point; coordinators can also open it directly by
adding `&review=final` to an existing private token URL.

While reviewing, click **Flag for redo** or press `F` to toggle the current
image's redo status. Flagged frame chips are amber, counts appear per scene and
for the full dataset, and flags from different reviewers are retained
independently. The server stores this audit queue in
`final_review_flags.json` at the annotation-run root. It does not modify the
frozen masks or invalidate their checksums.

### Correct masks flagged in the final browser

Freeze the current redo queue into a dedicated human editing assignment:

```bash
python -m egomask.redo prepare \
  --source-run /path/to/original_annotation_run \
  --final-export /path/to/current_frozen_export \
  --flags /path/to/original_annotation_run/final_review_flags.json \
  --output /path/to/flagged_correction_run \
  --reviewer annotator_1
```

Each flagged task opens with the current cleaned final mask as its editable
label. A magenta outline over the RGB and a standalone sidebar preview show the
untouched original 3D-component projection. That reference is copied into the
correction run, served read-only, and never enters the saved correction. Reset
label restores the cleaned final mask, not the 3D projection.

After every correction is explicitly completed, create a new frozen export:

```bash
python -m egomask.redo finalize \
  --correction-run /path/to/flagged_correction_run \
  --final-export /path/to/current_frozen_export \
  --output /path/to/corrected_frozen_export
```

Finalization refuses incomplete or corrupt corrections, replaces only the
flagged masks, records the human reviewer/revision/note and prior mask hash,
and regenerates the export checksum manifest.

Keep this terminal alive (a `tmux` session is convenient). The generated file
`ACCESS_LINKS.txt` contains one private URL path per annotator. Send each person
only their own link, appended to the server address, for example:

```text
http://SERVER_IP:8765/?token=PERSON_SPECIFIC_TOKEN
```

Tokens prevent accidental editing of another person's assignment. They are not
a substitute for Internet-facing authentication: use this only on a trusted
lab network. For remote access, prefer an SSH tunnel and keep the server bound
to localhost:

```bash
ssh -L 8765:localhost:8765 USER@SERVER
```

Then the annotator opens the assigned link as
`http://localhost:8765/?token=...`. Do not commit or post `ACCESS_LINKS.txt`.

## Annotator guide

Use a recent Chrome, Firefox, or Edge browser on a laptop/desktop.

1. Open your private link. Confirm that your name appears in the top bar.
2. Inspect the colored initialization at several overlay opacities. The RGB
   image is always the pre-action view.
3. Select the correct component color when a scene contains multiple instances.
4. Use **Brush** to add missing visible pixels and **Eraser** to remove spill.
   Use **Polygon** for long clean boundaries: click vertices, then press Enter.
5. Zoom for narrow stems, leaf edges, and fruit boundaries. Ctrl/Cmd + mouse
   wheel over the canvas zooms only the image. A normal wheel scrolls
   vertically; the scrollbars and Shift-drag also pan the image.
6. Press **Complete & next** only after checking every colored component.
   Use **Flag for review** when the target identity or boundary cannot be
   determined. Add a short note explaining why.

The tool autosaves shortly after every edit. “Saved” in the top bar confirms a
server-side atomic write. Closing a tab with unsaved edits triggers a warning.
Do not open the same private link in multiple tabs: revision checks prevent
silent overwrites, but one tab will be asked to reload.

Useful shortcuts:

| Key | Action |
|---|---|
| `B`, `E`, `P` | Brush, eraser, polygon |
| `1`–`8` | Select the corresponding component label |
| Hold `X` | Temporarily erase; release to return to the prior tool |
| `Ctrl/Cmd` + wheel | Zoom only the annotation image |
| `Ctrl/Cmd` + `+` / `-` / `0` | Image zoom in/out/reset while the canvas is active |
| `[` / `]` | Decrease/increase brush size |
| `Ctrl/Cmd+Z` | Undo |
| `Ctrl/Cmd+Y` or `Cmd+Shift+Z` | Redo |
| `Ctrl/Cmd+S` | Save draft |
| `Space` | Complete and open the next unfinished image |
| `F` | Flag for review and open the next unfinished image |
| `V` | Toggle boundary-only overlay |
| `R` | Reset the selected component to its initial projection |
| `Delete` / `Backspace` | Clear the selected component (Undo restores it) |
| `Enter` | Close the current polygon |
| `Esc` or right-click | Cancel the current polygon |
| `A` / `D` | Previous/next assigned image |

### Boundary policy

- Label only pixels visibly belonging to the plant component indicated by the
  initialization. The intended object is the pre-action structure that is gone
  after the intervention.
- Trace the visible silhouette. Do **not** fill pixels hidden behind another
  leaf, fruit, support, hand, tool, or other foreground object.
- Keep disconnected visible pieces when they are visibly parts of the same
  component separated by an occluder.
- Remove projected Gaussian spill from neighboring plant parts and background.
- When colored component projections overlap, assign the pixel to the component
  visibly in front. Never use overlapping labels; the label map stores one
  instance ID per pixel.
- If the assigned component is genuinely not visible, erase its projection,
  leave a note such as `component 2 not visible`, and complete the image.
- Do not expand a boundary merely because the object is likely to continue
  behind an occluder. These are image masks, not amodal masks.
- Flag identity ambiguity, severe blur, a missing image, or a grossly incorrect
  camera projection. Do not flag ordinary boundary cleanup.

Before starting the full run, all four annotators should jointly annotate the
same five practice images outside the production assignment and agree on edge,
occlusion, thin-stem, and empty-mask decisions. This short calibration prevents
systematic annotator drift. The production tasks themselves remain disjoint and
equally divided.

## Monitoring and quality control

The coordinator can check progress without stopping the server:

```bash
python -m egomask.status \
  --run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views
```

`needs_review` tasks must be adjudicated and saved as complete before the
default exporter will run. A sensible quality-control pass is:

1. resolve every flagged task with the annotator;
2. spot-check at least 5–10% of each person's masks, including every multi-
   component scene;
3. check empty masks and unusually tiny/large masks;
4. only then export a frozen dataset.

Export label maps, metadata, checksums, and optional per-component binary masks:

```bash
python -m egomask.export \
  --run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --output /path/to/data/EgoDelta/annotation_runs/image_masks_gt_wacv2027_v1 \
  --binary-components \
  --single-component
```

`--single-component` is the recommended primary-evaluation policy: it maps
every nonzero instance label to foreground label `1`. This reconciles
annotators who used different component IDs without changing the visible
silhouette. The live source masks and their original labels remain untouched,
and the export retains their component metadata as `source_components`.

### Recover polygons drawn over the starting mask

If an annotator drew filled polygons without first clearing the projected
starting mask, recover the human polygon regions before the single-component
export:

```bash
python -m egomask.polygon_recover \
  --run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --output /path/to/human_polygon_fills

python -m egomask.export \
  --run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --mask-overrides /path/to/human_polygon_fills \
  --output /path/to/human_polygon_export \
  --single-component \
  --binary-components \
  --solid-binary-masks
```

The recovery compares the saved component labels with the original projected
labels, reconstructs and fills the human-drawn polygon regions, and uses their
union as one foreground component. It defaults to annotators 1 and 2; masks
that were already substantially redrawn and all annotator-3/4 masks are kept
unchanged. The raw annotation PNGs are never edited. Check `audit.csv`, with
particular attention to `recover_and_review` or
`needs_human_review`, before freezing the export.

`--solid-binary-masks` applies the same final policy to every annotator: it
fills enclosed holes, closes narrow leaf-mask tool cuts, and removes isolated
brush or eraser fragments smaller than 16 pixels. Tomato masks use a gentler
closing size so nearby fruits are not bridged.

If a post-annotation audit finds a systematic projected-footprint versus tight-
silhouette policy split, generate a **candidate** multiview-consistent override
set instead of overwriting the raw annotations:

```bash
python -m egomask.multiview_unify \
  --run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --review-queue /path/to/review_queue.csv \
  --output /path/to/multiview_overrides

python -m egomask.export \
  --run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --mask-overrides /path/to/multiview_overrides \
  --output /path/to/candidate_export \
  --single-component
```

The multiview tool fits one Gaussian footprint per scene using the unflagged
tight human masks, measures leave-one-view-out agreement, transfers the fitted
projection to the projected-mask-style views, and replaces explicitly queued
corruption cases. Its `overrides.json` records every decision and source hash.
Treat the result only as a diagnostic proposal. It must not be substituted for
human GT. When every final mask must be human, create an independent second-pass
review initialized from the saved human annotations:

```bash
python -m egomask.adjudicate prepare \
  --source-run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --review-queue /path/to/review_queue.csv \
  --output /path/to/human_review_run
```

By default this selects all annotator-1/2 masks plus queued exceptions, collapses
their component IDs to a single foreground label, and assigns every task to a
different human reviewer with balanced workloads. Reviewers must correct the
visible RGB silhouette and explicitly complete every task. Automatic calibrated
masks are not loaded into this run.

After every second-pass task is complete, freeze the human-only export:

```bash
python -m egomask.adjudicate finalize \
  --source-run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --review-run /path/to/human_review_run \
  --override-output /path/to/human_review_overrides \
  --output /path/to/final_human_gt
```

Finalization refuses incomplete reviews, corrupt or stale masks, self-review,
and source-manifest changes. It records the human reviewer, revision, timestamp,
note, and source checksum for every replacement while leaving the raw run intact.

For an interim pilot after one assignment is complete, export only that
annotator without mixing in unfinished collaborators' masks:

```bash
python -m egomask.export \
  --run /path/to/data/EgoDelta/annotation_runs/wacv2027_masks_12views \
  --output /path/to/data/EgoDelta/annotation_runs/image_masks_gt_wacv2027_annotator1_pilot \
  --annotator annotator_1
```

The export contains:

```text
dataset.json                         # image, camera, instance, provenance metadata
annotations.csv                      # compact audit table
checksums.sha256                     # frozen-file integrity
label_maps/<scene>/<frame>.png       # canonical instance-label PNGs
binary_masks/<scene>/<frame>__component_XXX.png  # optional 0/255 masks
```

The exporter refuses a non-empty destination and refuses incomplete or flagged
tasks by default. `--allow-incomplete` exists only for intermediate debugging;
do not use such an export for paper results.

## Troubleshooting

- **The mask is offset everywhere in one image:** flag it and mention a camera
  projection problem. Do not manually translate the entire mask without review.
- **The browser says revision conflict:** another tab saved the same assignment.
  Close the duplicate tab and reload.
- **A save fails:** keep the tab open, verify the server is running and the run
  disk is writable, then press Save draft again.
- **Preparation is slow:** use `--workers 4`. The source 2DGS PLY is read once
  per active scene worker; avoid excessive workers on a memory-constrained host.
- **Port 8765 is busy:** choose another `--port` and update the shared URLs.
- **Need a smoke test:** add `--limit-scenes 1 --workers 1` and use a new output
  directory.
