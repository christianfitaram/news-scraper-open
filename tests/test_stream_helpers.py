"""Additional tests for scraper stream helper behavior."""
from __future__ import annotations

from types import SimpleNamespace

from news_crawler.scrapers import streams


def test_should_skip_prefetch_tracks_unprocessed_links(monkeypatch):
    calls: dict[str, object] = {}

    class _Repo:
        def is_processed(self, url):
            calls["checked"] = url
            return False

        def upsert_link(self, url, extra):
            calls["upsert"] = (url, extra["last_seen_source"])

    monkeypatch.setattr(streams, "_get_link_pool_repo", lambda: _Repo())

    assert (
        streams._should_skip_prefetch(
            "https://example.com/a",
            source="bbc",
            check_link_pool=True,
            track_links=True,
        )
        is False
    )
    assert calls["checked"] == "https://example.com/a"
    assert calls["upsert"] == ("https://example.com/a", "bbc")


def test_should_skip_prefetch_skips_processed_links(monkeypatch):
    class _Repo:
        def is_processed(self, url):
            del url
            return True

        def upsert_link(self, url, extra):
            raise AssertionError("processed links should not be upserted")

    monkeypatch.setattr(streams, "_get_link_pool_repo", lambda: _Repo())

    assert streams._should_skip_prefetch(
        "https://example.com/a",
        source="bbc",
        check_link_pool=True,
        track_links=True,
    )


def test_yield_rss_with_extracted_text_skips_invalid_entries(monkeypatch):
    feed = SimpleNamespace(
        entries=[
            {"title": "", "link": "https://example.com/missing-title"},
            {"title": "No link"},
            {"title": "Good", "link": "https://example.com/good"},
            {"title": "No text", "link": "https://example.com/no-text"},
        ]
    )

    monkeypatch.setattr(streams.feedparser, "parse", lambda url: feed)
    monkeypatch.setattr(
        streams,
        "fetch_and_extract",
        lambda url: "extracted body" if url.endswith("/good") else "",
    )

    articles = list(
        streams._yield_rss_with_extracted_text(
            {"world": "https://feed.example/rss"},
            source="guardian",
            check_link_pool=False,
            track_links=False,
        )
    )

    assert len(articles) == 1
    assert articles[0]["title"] == "Good"
    assert articles[0]["text"] == "extracted body"
    assert articles[0]["source"] == "guardian"


def test_wsj_stream_uses_summary_without_fetching(monkeypatch):
    feed = SimpleNamespace(
        entries=[
            {
                "title": "WSJ headline",
                "link": "https://wsj.example/a",
                "summary": "Summary text",
            }
        ]
    )

    monkeypatch.setattr(streams.feedparser, "parse", lambda url: feed)
    monkeypatch.setattr(
        streams,
        "fetch_and_extract",
        lambda url: (_ for _ in ()).throw(AssertionError("WSJ should use summary")),
    )

    articles = list(streams.scrape_wsj_stream(check_link_pool=False, track_links=False))

    assert articles[0]["title"] == "WSJ headline"
    assert articles[0]["text"] == "Summary text"
    assert articles[0]["source"] == "wsj"

