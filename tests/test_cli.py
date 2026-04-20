"""Tests for the Typer CLI surface."""
from __future__ import annotations

from typer.testing import CliRunner

from news_crawler import cli as cli_module

runner = CliRunner()


def test_run_command_passes_limit_and_dry_run(monkeypatch):
    calls: dict[str, object] = {}

    class _StateManager:
        def reset(self):
            calls["reset"] = True

        def get_last_batch_id(self):
            return "batch-1"

    class _Orchestrator:
        def __init__(self, *, state_manager, dry_run):
            calls["state_manager"] = state_manager
            calls["dry_run"] = dry_run

        def run(self, limit):
            calls["limit"] = limit
            return {
                "articles_processed": 2,
                "articles_skipped": 1,
                "articles_failed": 0,
                "duration_seconds": 1.25,
            }

    monkeypatch.setattr(cli_module, "setup_logging", lambda level: calls.setdefault("log_level", level))
    monkeypatch.setattr(cli_module, "validate_config", lambda **kwargs: calls.setdefault("validate", kwargs))
    monkeypatch.setattr(cli_module, "StateManager", _StateManager)
    monkeypatch.setattr(cli_module, "PipelineOrchestrator", _Orchestrator)

    result = runner.invoke(cli_module.app, ["run", "--dry-run", "--limit", "5", "--resume", "--reset-state"])

    assert result.exit_code == 0
    assert calls["dry_run"] is True
    assert calls["limit"] == 5
    assert calls["reset"] is True
    assert calls["validate"] == {"require_db": False, "require_genai": False}
    assert "Pipeline completed successfully" in result.output
    assert "Articles processed: 2" in result.output


def test_run_command_handles_keyboard_interrupt(monkeypatch):
    class _StateManager:
        pass

    class _Orchestrator:
        def __init__(self, *, state_manager, dry_run):
            del state_manager, dry_run

        def run(self, limit):
            del limit
            raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "setup_logging", lambda level: None)
    monkeypatch.setattr(cli_module, "validate_config", lambda **kwargs: None)
    monkeypatch.setattr(cli_module, "StateManager", _StateManager)
    monkeypatch.setattr(cli_module, "PipelineOrchestrator", _Orchestrator)

    result = runner.invoke(cli_module.app, ["run", "--dry-run"])

    assert result.exit_code == 1
    assert "Pipeline interrupted by user" in result.output


def test_status_command_prints_state(monkeypatch):
    class _StateManager:
        def get_last_batch_id(self):
            return "batch-42"

        def get_pipeline_stats(self):
            return {
                "total_articles_processed": 12,
                "total_batches": 3,
                "last_run": "2026-04-20T00:00:00+00:00",
            }

    monkeypatch.setattr(cli_module, "StateManager", _StateManager)

    result = runner.invoke(cli_module.app, ["status"])

    assert result.exit_code == 0
    assert "Last batch ID: batch-42" in result.output
    assert "Total articles: 12" in result.output
    assert "Total batches: 3" in result.output


def test_reset_command_confirms_and_resets(monkeypatch):
    calls: dict[str, bool] = {}

    class _StateManager:
        def reset(self):
            calls["reset"] = True

    monkeypatch.setattr(cli_module, "StateManager", _StateManager)

    result = runner.invoke(cli_module.app, ["reset"], input="y\n")

    assert result.exit_code == 0
    assert calls["reset"] is True
    assert "State reset successfully" in result.output

