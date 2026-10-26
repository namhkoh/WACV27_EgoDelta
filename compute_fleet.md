# compute-node-2: WACV final-720 runs

compute-node-2 should run only its assigned 3DGS-CD partition. GS-DIFF, MV3DCD, and
EgoDelta are already complete on all 720 images.

The global 3DGS-CD partition is fixed as follows:

- compute-node-1: workers 0–23 of 40.
- compute-node-2: workers 24–39 of 40, with eight workers on each 96 GB GPU.

Do not run `run_all_scenes.sh 3dgscd` directly on compute-node-2; that would duplicate
scenes already running on compute-node-1.

## Launch

From the shared EgoDelta checkout on compute-node-2:

```bash
cd /path/to/data/EgoDelta
git switch redacted/WACV
bash baseline/scene_change/scripts/launch_compute_fleet.sh
```

The launcher performs a two-step Blackwell/CUDA compatibility test on GPU 0
before starting any benchmark job. It uses CUDA 9.0 PTX for driver-side forward
JIT with the official 3DGS-CD environment. If the preflight fails, no compute-node-2
benchmark workers are launched; preserve the error output and report it before
trying a different environment.

After a successful preflight, the launcher creates the persistent tmux session
`wacv-3dgscd-compute-node-2` and returns to the shell.

## Monitor

```bash
tmux attach -t wacv-3dgscd-compute-node-2
```

Detach without stopping jobs with `Ctrl-b`, then `d`.

In another terminal:

```bash
watch -n 2 nvidia-smi
```

compute-node-2-local worker logs are written to:

```text
/tmp/egodelta-3dgscd-final720-compute-node-2/logs/
```

Completed worker logs are copied to the shared durable directory:

```text
baseline/scene_change/runs/logs/final_720_20260817/fleet/
```

Check global completion across both servers with:

```bash
find baseline/scene_change/runs/predictions/3dgscd \
  -mindepth 2 -maxdepth 2 -type f -name metadata.json | wc -l
```

The target is 60. compute-node-1's finalizer automatically runs all final evaluations
and creates the Figure-3-style PNG/PDF after coverage reaches 60/60.
