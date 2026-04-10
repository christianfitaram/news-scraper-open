# Contributing

Thanks for contributing to NewsFeeder-IA.

## Scope

This project is a modular news scraping and enrichment pipeline. Contributions are welcome in these areas:

- scraper reliability
- tests and CI
- documentation
- MongoDB repository improvements
- provider integrations for the existing Gemini/Ollama architecture

## Development Setup

Recommended setup:

```bash
poetry install
poetry shell
```

Alternative setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock
```

Create your local environment file:

```bash
cp .env.example .env
```

## Running Tests

```bash
pytest
```

Focused test runs:

```bash
pytest tests/test_orchestrator.py
pytest tests/test_summarizer_chunking.py
```

## Coding Guidelines

- Keep changes scoped and easy to review.
- Prefer extending `news_crawler/` instead of adding compatibility shims.
- Reuse shared helpers in `news_crawler/scrapers/streams.py`, `news_crawler/processors/_hf_common.py`, and `news_crawler/providers/_entity_common.py`.
- Add tests for behavior changes, especially for provider selection, chunking, and scraper registration.
- Avoid committing local artifacts such as `.env`, `models/`, `outputs/`, `htmlcov/`, or state files.

## Scraping Notes

- Respect the terms of service and access policies of upstream sources.
- Expect source markup to change over time; scraper contributions should fail gracefully.
- If a source requires Selenium, document why a static parser is insufficient.

## Pull Requests

- Open a focused PR with a clear title and summary.
- Include reproduction steps for bug fixes.
- Mention any environment variables, data migrations, or backward-compatibility impacts.
- Update `README.md` when behavior or setup changes.
