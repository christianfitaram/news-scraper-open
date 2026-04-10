"""CLI entry point for News Crawler AI."""
from __future__ import annotations

import logging
import sys
from typing import Optional

import typer
from rich.console import Console

from news_crawler.core.config import APP_CONFIG, validate_config
from news_crawler.core.orchestrator import PipelineOrchestrator
from news_crawler.core.state import StateManager
from news_crawler.utils.logging import setup_logging

app = typer.Typer(name="news-crawler", help="News Crawler AI - scraping and classification pipeline")
console = Console()


@app.command()
def run(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max articles to process"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without writing to DB"),
    resume: bool = typer.Option(False, "--resume", help="Resume from last state"),
    reset_state: bool = typer.Option(False, "--reset-state", help="Reset state before running"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Run the complete scraping and classification pipeline."""
    log_level = "DEBUG" if verbose else APP_CONFIG.log_level
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    console.print("[bold blue]News-Crawler-AI Pipeline[/bold blue]")
    console.print(f"Dry run: {dry_run}")
    console.print(f"Limit: {limit or 'unlimited'}")
    console.print()

    try:
        require_genai = APP_CONFIG.enable_genai and not APP_CONFIG.enable_ollama and not dry_run
        validate_config(require_db=not dry_run, require_genai=require_genai)

        state_manager = StateManager()
        if reset_state:
            state_manager.reset()
            console.print("[yellow]State reset[/yellow]")

        if resume:
            last_batch = state_manager.get_last_batch_id()
            if last_batch:
                console.print(f"[green]Resuming from batch: {last_batch}[/green]")
            else:
                console.print("[yellow]No previous state found[/yellow]")

        orchestrator = PipelineOrchestrator(state_manager=state_manager, dry_run=dry_run)
        stats = orchestrator.run(limit=limit)

        console.print()
        console.print("[bold green]Pipeline completed successfully![/bold green]")
        console.print(f"Articles processed: {stats['articles_processed']}")
        console.print(f"Articles skipped: {stats['articles_skipped']}")
        console.print(f"Articles failed: {stats['articles_failed']}")
        console.print(f"Duration: {stats.get('duration_seconds', 0):.2f}s")
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        console.print(f"\n[bold red]Pipeline failed: {exc}[/bold red]")
        sys.exit(1)


@app.command()
def status():
    """Show pipeline status and statistics."""
    state_manager = StateManager()

    console.print("[bold]Pipeline Status[/bold]")
    console.print()

    last_batch = state_manager.get_last_batch_id()
    if last_batch:
        console.print(f"Last batch ID: {last_batch}")
    else:
        console.print("No batches run yet")

    stats = state_manager.get_pipeline_stats()
    console.print(f"Total articles: {stats['total_articles_processed']}")
    console.print(f"Total batches: {stats['total_batches']}")
    if stats.get("last_run"):
        console.print(f"Last run: {stats['last_run']}")


@app.command()
def reset():
    """Reset pipeline state."""
    state_manager = StateManager()
    confirmed = typer.confirm("Are you sure you want to reset all state?")
    if confirmed:
        state_manager.reset()
        console.print("[green]State reset successfully[/green]")
    else:
        console.print("[yellow]Cancelled[/yellow]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
