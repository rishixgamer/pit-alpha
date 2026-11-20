import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from quantcheck.pit_alpha.cli import app

runner = CliRunner()


def test_example_success_reproduces_both_reports_and_rejects_conflicting_output(
    tmp_path: Path,
) -> None:
    assert runner.invoke(app, ["example", "--directory", str(tmp_path)]).exit_code == 0
    config = tmp_path / "example.yaml"
    command = ["run", "--config", str(config)]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    output = tmp_path / "reports"
    report = (output / "report.json").read_bytes()
    markdown = (output / "report.md").read_bytes()
    document = json.loads(report)
    selection = document["selection"]
    selection_check = selection["check"]
    assert document["schema_version"] == 1
    assert len(document["folds"]) == 5
    assert selection_check["holdout_periods"] == 150
    assert selection_check["passes"]
    assert str(selection_check["adjusted_p"]).encode() in markdown
    assert selection["selected_id"].encode() in markdown
    assert str(document["metrics"]["combined"]["ic"]["value"]).encode() in markdown
    assert runner.invoke(app, command).exit_code == 0
    assert (output / "report.json").read_bytes() == report
    assert (output / "report.md").read_bytes() == markdown

    config.write_text(config.read_text().replace("seed: 812", "seed: 813"))
    conflict = runner.invoke(app, command)
    assert conflict.exit_code == 1
    assert "already contains different results" in conflict.output
    assert (output / "report.json").read_bytes() == report
    assert (output / "report.md").read_bytes() == markdown


def test_invalid_config_schema_and_empty_folds_publish_no_reports(tmp_path: Path) -> None:
    assert runner.invoke(app, ["example", "--directory", str(tmp_path)]).exit_code == 0
    config = tmp_path / "example.yaml"
    text = config.read_text()
    command = ["run", "--config", str(config)]
    for invalid, message in [
        (text.replace("schema_version: 1", "schema_version: 7"), "schema_version"),
        (text.replace("schema_version: 1", "schema_version: true"), "schema_version"),
        (text.replace("schema_version: 1", "schema_version: 1.0"), "schema_version"),
        (text + "schema_version: 1\n", "unique keys"),
        (
            text.replace("train_seconds: 5184000", "train_seconds: 315576000"),
            "complete walk-forward",
        ),
    ]:
        config.write_text(invalid)
        result = runner.invoke(app, command)
        assert result.exit_code == 1
        assert message in result.output
        assert not (tmp_path / "reports").exists()
    config.write_text(text)
    source = tmp_path / "observations.parquet"
    pl.read_parquet(source).drop("available_ts").write_parquet(source)
    result = runner.invoke(app, command)
    assert result.exit_code == 1
    assert "available_ts" in result.output
    assert not (tmp_path / "reports").exists()
