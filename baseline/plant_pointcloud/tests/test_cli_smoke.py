from __future__ import annotations

from typer.testing import CliRunner

from tomato_cutpoints.cli import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "3D removal target detection" in result.output


def test_cli_synthetic_demo_runs(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["synthetic-demo", "--task", "harvest", "--out-dir", str(tmp_path), "--no-perturb-post"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "targets.json").exists()
