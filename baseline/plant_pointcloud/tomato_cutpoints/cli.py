from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from tomato_cutpoints import __version__
from tomato_cutpoints.eval.metrics import evaluate_files, format_metrics_table
from tomato_cutpoints.pipeline import run_pipeline
from tomato_cutpoints.synthetic import write_synthetic_trial
from tomato_cutpoints.utils.logging import setup_logging
from tomato_cutpoints.viz.open3d_viz import view_run_plys

app = typer.Typer(help="3D removal target detection from Project Aria MPS semi-dense point clouds.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version_callback, help="Show package version."),
) -> None:
    del version


@app.command("run")
def run_cmd(
    task: str = typer.Option(..., help="Task: harvest or deleaf."),
    pre_points: Path = typer.Option(..., help="Pre-action slam/semidense_points.csv.gz."),
    post_points: Path = typer.Option(..., help="Post-action slam/semidense_points.csv.gz."),
    out_dir: Path = typer.Option(..., help="Output run directory."),
    config: Optional[Path] = typer.Option(None, help="YAML config path."),
    pre_observations: Optional[Path] = typer.Option(None, help="Pre-action semidense_observations.csv.gz."),
    post_observations: Optional[Path] = typer.Option(None, help="Post-action semidense_observations.csv.gz."),
    stable_roi_json: Optional[Path] = typer.Option(None, help="ROI JSON with crop/exclude/stable boxes."),
    refine_icp: bool = typer.Option(False, help="Refine post-to-pre alignment with ICP."),
    post_to_pre_transform: Optional[Path] = typer.Option(None, help="Text file containing a 4 x 4 post-to-pre transform."),
    strict: bool = typer.Option(False, help="Fail instead of falling back if optional M3C2 fails."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    logger = setup_logging(out_dir, verbose=verbose)
    task = task.lower()
    if task not in {"harvest", "deleaf"}:
        raise typer.BadParameter("task must be one of: harvest, deleaf")
    logger.info("Running %s pipeline", task)
    run_pipeline(
        task=task,
        pre_points=pre_points,
        post_points=post_points,
        pre_observations=pre_observations,
        post_observations=post_observations,
        out_dir=out_dir,
        config_path=config,
        stable_roi_json=stable_roi_json,
        refine_icp=refine_icp,
        post_to_pre_transform=post_to_pre_transform,
        strict=strict,
        verbose=verbose,
    )
    typer.echo(f"Wrote run outputs to {out_dir}")


@app.command("evaluate")
def evaluate_cmd(
    pred: Path = typer.Option(..., help="Predicted targets.json."),
    gt: Path = typer.Option(..., help="Ground-truth JSON."),
) -> None:
    metrics = evaluate_files(pred, gt)
    typer.echo(format_metrics_table(metrics))
    typer.echo(f"Wrote {pred.parent / 'eval_metrics.json'}")


@app.command("synthetic-demo")
def synthetic_demo_cmd(
    task: str = typer.Option("harvest", help="Task: harvest or deleaf."),
    out_dir: Path = typer.Option(..., help="Output directory for synthetic inputs and run outputs."),
    seed: int = typer.Option(13, help="Deterministic random seed."),
    no_perturb_post: bool = typer.Option(False, help="Do not apply a small synthetic post-frame perturbation."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    logger = setup_logging(out_dir, verbose=verbose)
    task = task.lower()
    if task not in {"harvest", "deleaf"}:
        raise typer.BadParameter("task must be one of: harvest, deleaf")
    paths = write_synthetic_trial(out_dir, task, seed=seed, perturb_post=not no_perturb_post)
    run_dir = out_dir / "run"
    logger.info("Running synthetic %s demo", task)
    run_pipeline(
        task=task,
        pre_points=paths["pre_points"],
        post_points=paths["post_points"],
        pre_observations=paths["pre_observations"],
        post_observations=paths["post_observations"],
        stable_roi_json=paths["roi"],
        out_dir=run_dir,
        config_path=paths["config"],
        refine_icp=True,
        strict=False,
        verbose=verbose,
    )
    typer.echo(f"Wrote synthetic run outputs to {run_dir}")
    typer.echo(f"Ground truth: {paths['gt']}")


@app.command("view")
def view_cmd(
    run_dir: Path = typer.Option(..., help="Run directory containing plys/."),
) -> None:
    view_run_plys(run_dir)


if __name__ == "__main__":
    app()
