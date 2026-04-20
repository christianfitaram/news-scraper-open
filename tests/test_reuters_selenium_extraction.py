"""Tests for Reuters Selenium URL extraction helpers."""
from __future__ import annotations

from news_crawler.scrapers import streams
from news_crawler.scrapers.streams import extract_reuters_article_candidates


def test_extract_reuters_candidates_prefers_article_urls_from_json_ld():
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context":"https://schema.org",
          "mainEntity":{
            "@type":"ItemList",
            "itemListElement":[
              {"@type":"ListItem","url":"https://www.reuters.com/world/europe/example-story-2026-04-19/"},
              {"@type":"ListItem","url":"https://www.reuters.com/world/"}
            ]
          }
        }
        </script>
      </head>
      <body>
        <a href="/world/europe/example-story-2026-04-19/">Example Story</a>
        <a href="/video/watch-this">Video Link</a>
      </body>
    </html>
    """

    candidates = extract_reuters_article_candidates(html)
    urls = [url for url, _ in candidates]

    assert "https://www.reuters.com/world/europe/example-story-2026-04-19/" in urls
    assert "https://www.reuters.com/world/" not in urls
    assert "https://www.reuters.com/video/watch-this" not in urls


def test_extract_reuters_candidates_dedupes_urls():
    html = """
    <html>
      <body>
        <a href="https://www.reuters.com/world/example-title-2026-04-19/">One</a>
        <a href="https://www.reuters.com/world/example-title-2026-04-19/">Two</a>
      </body>
    </html>
    """

    candidates = extract_reuters_article_candidates(html)

    assert len(candidates) == 1
    assert candidates[0][0] == "https://www.reuters.com/world/example-title-2026-04-19/"


def test_extract_reuters_candidates_from_escaped_js_payload_urls():
    html = r"""
    <html>
      <head>
        <script>
          window.__STATE__ = {
            "cards": [
              {
                "url": "https:\/\/www.reuters.com\/markets\/europe\/example-markets-story-2026-04-19\/"
              }
            ]
          };
        </script>
      </head>
    </html>
    """

    candidates = extract_reuters_article_candidates(html)
    urls = [url for url, _ in candidates]

    assert "https://www.reuters.com/markets/europe/example-markets-story-2026-04-19/" in urls


def test_reuters_selenium_stream_falls_back_to_rss_when_blocked(monkeypatch):
    class _BlockedDriver:
        current_url = "https://www.reuters.com/"
        title = "reuters.com"
        page_source = "<html><body>verify you are human</body></html>"

        def get(self, url: str) -> None:
            self.current_url = url

        def quit(self) -> None:
            return None

    def _fake_build_driver(*, headless: bool = True):  # noqa: ARG001
        return _BlockedDriver()

    def _fake_rss_stream(*, check_link_pool: bool = True, track_links: bool = True):  # noqa: ARG001
        yield {
            "title": "Fallback Reuters Story",
            "url": "https://www.reuters.com/world/example-story-2026-04-19/",
            "text": "x" * 300,
            "source": "reuters",
        }

    monkeypatch.setattr(streams, "build_chrome_driver", _fake_build_driver)
    monkeypatch.setattr(streams, "scrape_reuters_stream", _fake_rss_stream)
    monkeypatch.setattr(streams.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(streams, "REUTERS_SECTION_URLS", {"home": "https://www.reuters.com/"})

    articles = list(
        streams.scrape_reuters_selenium_stream(
            check_link_pool=False,
            track_links=False,
        )
    )

    assert len(articles) == 1
    assert articles[0]["source"] == "reuters-selenium"
    assert articles[0]["title"] == "Fallback Reuters Story"
