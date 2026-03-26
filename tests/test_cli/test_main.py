from typer.testing import CliRunner

runner = CliRunner()


def test_cli_help():
    from litscribe.cli.main import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LitScribe" in result.output


def test_cli_review_help():
    from litscribe.cli.main import app

    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    assert "question" in result.output.lower() or "QUESTION" in result.output


def test_cli_skills_help():
    from litscribe.cli.main import app

    result = runner.invoke(app, ["skills", "--help"])
    assert result.exit_code == 0


def test_cli_config_help():
    from litscribe.cli.main import app

    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0
