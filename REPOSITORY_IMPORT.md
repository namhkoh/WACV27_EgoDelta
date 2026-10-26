# Import into a new GitHub repository

Extract the ZIP and open a terminal inside `EgoDelta_WACV27_camera_ready`,
the directory containing this file. Create an empty GitHub repository, then:

~~~bash
git init -b main
git add .
git commit -m "Initial WACV 2027 code release"
git remote add origin <new-repository-url>
git push -u origin main
~~~

The ZIP contains no `.git` directory or previous commit history. Git LFS is not
needed for the bundled checkpoints: their pointers are resolved and the old
LFS filters removed. Ignore rules retain the checkpoint arrays, figures, and
source fixtures when running `git add .`.

## Package scope

- Revision: `1659525f888125cfa8bef706c57da1dc0cbe329f` on
  `redacted/WACV27-camera-ready`.
- Includes tracked source, configurations, tests, documentation,
  reproducibility evidence, and all tracked policy checkpoint arrays.
- Includes exact pinned sources for diff-surfel-rasterization, simple-knn,
  and nested GLM. They are ordinary source directories; `.gitmodules` is omitted.
  Their revisions appear in `RELEASE_INFO.json`.
- Omits `robot/policy/policy_action_library/`, the recorded demonstration
  dataset and its indexes. Supply training data separately.
- Does not copy untracked files, working-tree changes, local recordings,
  external datasets, generated caches, vendor SDK installations, or Git history.
- Preserves `target_extraction/old/` for legacy Stage-2 and reproduction dependencies.
- Preserves the branch's metadata-only legacy policy folders as supplied.
  The four checkpoint-array files present in the source revision are included.
  No absent checkpoints or benchmark caches have been fabricated.

`RELEASE_EXCLUSIONS.json` lists deliberately omitted tracked paths.
`RELEASE_INFO.json` records source revision, dependency pins, resolved LFS
objects, and packaging-only edits. `MANIFEST.sha256` covers all other files.

## Running the code

Start with the component links in `README.md`. Reconstruction dependencies
are specified in `environment.yml`; build CUDA extensions for your machine.
The retained Stage-3 method consumes frozen Stage-2 caches and ground truth
not supplied by this archive. RBY1 execution needs the vendor SDK, robot/camera
access, and your calibration. Examples may reference the original laboratory's
filesystem paths.

Archive integrity and fresh Git import are verified during packaging. GPU
training, benchmark reproduction, and hardware execution are separate checks.
