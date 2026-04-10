"""Shared scraper stream implementations used by the package and legacy shims."""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup

from news_crawler.core.config import NEWSAPI_CONFIG
from news_crawler.utils.text import fetch_and_extract

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

DW_URL = "https://www.dw.com/en/top-stories/s-9097"
DW_LINK_PATTERNS = [
    r"^https?://(www\.)?dw\.com/en/.+/(a|video|g)-\d+",
    r"^https?://(www\.)?dw\.com/en/.+/(a|video|g)-[A-Za-z0-9\-]+",
    r"^/en/.+/(a|video|g)-\d+",
    r"^/en/.+/(a|video|g)-[A-Za-z0-9\-]+",
]
COOKIE_ACCEPT_TEXTS = [
    "accept",
    "accept all",
    "agree",
    "ok",
    "allow",
    "aceptar",
    "aceitar",
    "zustimmen",
    "akzeptieren",
    "accepter",
]
COOKIE_BUTTON_XPATHS = [
    "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
    "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
    "//button[contains(@class,'cookie') or contains(@id,'cookie')]",
]


def _scraped_at() -> datetime:
    return datetime.now(timezone.utc)


def _article(title: str, url: str, text: str, source: str) -> Dict[str, Any]:
    return {
        "title": title.strip(),
        "url": url,
        "text": text,
        "source": source,
        "scraped_at": _scraped_at(),
    }


def _request(url: str, timeout: int = 10) -> requests.Response:
    response = requests.get(url, timeout=timeout, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    return response


def _parse_html(url: str, timeout: int = 10) -> BeautifulSoup | None:
    try:
        return BeautifulSoup(_request(url, timeout=timeout).text, "html.parser")
    except Exception as exc:
        logger.warning("Failed to fetch HTML from %s: %s", url, exc)
        return None


def _yield_rss_with_extracted_text(
    feeds: Dict[str, str],
    *,
    source: str,
) -> Iterator[Dict[str, Any]]:
    for category, rss_url in feeds.items():
        try:
            feed = feedparser.parse(rss_url)
        except Exception as exc:
            logger.warning("Failed to parse %s RSS feed %s: %s", source, category, exc)
            continue

        for entry in feed.entries:
            url = entry.get("link")
            title = (entry.get("title") or "").strip()
            if not url or not title:
                continue
            text = fetch_and_extract(url)
            if not text:
                continue
            yield _article(title, url, text, source)


def build_chrome_driver(headless: bool = True) -> Any:
    """Create a Chrome driver with safe defaults for headless/systemd environments."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    user_data_dir = os.getenv("CHROME_USER_DATA_DIR", "/tmp/chrome-user-data")
    chrome_options.add_argument(f"--user-data-dir={user_data_dir}")

    chrome_bin = os.getenv("CHROME_BIN")
    if chrome_bin:
        chrome_options.binary_location = chrome_bin

    driver_path = os.getenv("CHROMEDRIVER_PATH")
    if driver_path:
        service = Service(driver_path)
    else:
        driver_version = os.getenv("CHROMEDRIVER_VERSION")
        try:
            service = Service(ChromeDriverManager(version=driver_version).install())
        except TypeError:
            service = Service(ChromeDriverManager().install())

    return webdriver.Chrome(service=service, options=chrome_options)


def _resolve_chrome_binary() -> str:
    env_path = os.getenv("CHROME_BINARY")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        "/snap/bin/chromium",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError("Chrome/Chromium binary not found")


def _resolve_chromedriver_path() -> str:
    env_path = os.getenv("CHROMEDRIVER_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    for candidate in ["/usr/bin/chromedriver", shutil.which("chromedriver")]:
        if candidate and os.path.exists(candidate):
            return candidate

    from webdriver_manager.chrome import ChromeDriverManager

    return ChromeDriverManager().install()


def _make_dw_profile_dir() -> str:
    base = os.path.expanduser("~/snap/chromium/common/selenium-profiles")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="dw-", dir=base)


def _build_dw_driver(headless: bool = True) -> tuple[Any, str]:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service

    options = webdriver.ChromeOptions()
    options.binary_location = _resolve_chrome_binary()

    if headless:
        options.add_argument("--headless=new")
    else:
        options.add_argument("--start-maximized")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=0")
    options.add_argument("--window-size=1920,1080")

    profile_dir = _make_dw_profile_dir()
    options.add_argument(f"--user-data-dir={profile_dir}")
    options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(_resolve_chromedriver_path()),
        options=options,
    )
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(30)
    return driver, profile_dir


def _try_click(element: Any) -> bool:
    try:
        element.click()
        return True
    except Exception:
        try:
            element._parent.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            return False


def _dismiss_cookie_modal(driver: Any) -> bool:
    from selenium.webdriver.common.by import By

    time.sleep(1)
    for xpath in COOKIE_BUTTON_XPATHS:
        for element in driver.find_elements(By.XPATH, xpath):
            label = (element.text or "").lower()
            if any(token in label for token in COOKIE_ACCEPT_TEXTS) and element.is_displayed():
                if _try_click(element):
                    time.sleep(0.5)
                    return True
    return False


def _extract_dw_links(driver: Any) -> list[str]:
    from selenium.webdriver.common.by import By

    links = set()
    for anchor in driver.find_elements(By.TAG_NAME, "a"):
        href = anchor.get_attribute("href")
        if not href:
            continue
        href = href.strip()
        if href.startswith("/"):
            href = f"https://www.dw.com{href}"
        for pattern in DW_LINK_PATTERNS:
            if re.search(pattern, href):
                links.add(href)
                break
    return sorted(links)


def get_title_from_dw_url(url: str) -> str:
    soup = _parse_html(url)
    if soup is None:
        return "DW Article"
    title_tag = soup.find("h1")
    return title_tag.get_text(strip=True) if title_tag else "DW Article"


def crawl_dw_links(headless: bool = True) -> list[str]:
    try:
        driver, profile_dir = _build_dw_driver(headless=headless)
    except Exception as exc:
        logger.warning("Cannot start DW Selenium driver: %s", exc)
        return []

    try:
        driver.get(DW_URL)
        time.sleep(2)
        _dismiss_cookie_modal(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        return _extract_dw_links(driver)
    finally:
        try:
            driver.quit()
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)


def scrape_bbc_stream(*, check_link_pool: bool = True, track_links: bool = True) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    soup = _parse_html("https://www.bbc.com/news")
    if soup is None:
        return

    seen_urls: set[str] = set()
    for headline in soup.select("a[href^='/news'] h2"):
        title = headline.get_text(strip=True)
        parent = headline.find_parent("a")
        href = parent.get("href") if parent else ""
        full_url = urljoin("https://www.bbc.com", href)
        if not title or not full_url or full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        text = fetch_and_extract(full_url)
        if not text:
            continue
        yield _article(title, full_url, text, "bbc-news")


def scrape_cnn_stream(*, check_link_pool: bool = True, track_links: bool = True) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    soup = _parse_html("https://edition.cnn.com/world")
    if soup is None:
        return

    seen_urls: set[str] = set()
    for link in soup.select("a[data-link-type='article']"):
        href = link.get("href", "")
        if not href:
            continue
        full_url = urljoin("https://edition.cnn.com", href)
        title_tag = link.select_one(".container__headline-text, [data-editable='headline']")
        if title_tag is None:
            continue
        title = title_tag.get_text(strip=True)
        if not title or full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        text = fetch_and_extract(full_url)
        if not text:
            continue
        yield _article(title, full_url, text, "cnn")


def scrape_wsj_stream(*, check_link_pool: bool = True, track_links: bool = True) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    try:
        feed = feedparser.parse("https://feeds.a.dj.com/rss/RSSWorldNews.xml")
    except Exception as exc:
        logger.warning("Failed to parse WSJ RSS feed: %s", exc)
        return

    for entry in feed.entries:
        url = entry.get("link")
        title = (entry.get("title") or "").strip()
        summary = (entry.get("summary") or "").strip()
        if url and title and summary:
            yield _article(title, url, summary, "wsj")


def scrape_aljazeera(*, check_link_pool: bool = True, track_links: bool = True) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    try:
        feed = feedparser.parse("https://www.aljazeera.com/xml/rss/all.xml")
    except Exception as exc:
        logger.warning("Failed to parse Al Jazeera RSS feed: %s", exc)
        return

    for entry in feed.entries:
        url = entry.get("link")
        title = (entry.get("title") or "").strip()
        if not url or not title:
            continue
        text = fetch_and_extract(url)
        if not text:
            continue
        yield _article(title, url, text, "aljazeera")


def scrape_dw_stream(*, check_link_pool: bool = True, track_links: bool = True) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    for link in crawl_dw_links(headless=True):
        text = fetch_and_extract(link)
        if not text:
            continue
        yield _article(get_title_from_dw_url(link), link, text, "dw")


def scrape_guardian_stream(*, check_link_pool: bool = True, track_links: bool = True) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    feeds = {
        "world": "https://www.theguardian.com/world/rss",
        "business": "https://www.theguardian.com/business/rss",
        "technology": "https://www.theguardian.com/technology/rss",
        "science": "https://www.theguardian.com/science/rss",
        "environment": "https://www.theguardian.com/environment/rss",
        "politics": "https://www.theguardian.com/politics/rss",
    }
    yield from _yield_rss_with_extracted_text(feeds, source="the-guardian")


def scrape_reuters_stream(*, check_link_pool: bool = True, track_links: bool = True) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    feeds = {
        "world": "https://www.reuters.com/rssFeed/worldNews",
        "business": "https://www.reuters.com/rssFeed/businessNews",
        "technology": "https://www.reuters.com/rssFeed/technologyNews",
        "sports": "https://www.reuters.com/rssFeed/sportsNews",
        "entertainment": "https://www.reuters.com/rssFeed/entertainmentNews",
    }
    yield from _yield_rss_with_extracted_text(feeds, source="reuters")


def scrape_guardian_selenium_stream(
    *,
    check_link_pool: bool = True,
    track_links: bool = True,
) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    from selenium.webdriver.common.by import By

    sections = {
        "world": "https://www.theguardian.com/world",
        "business": "https://www.theguardian.com/business",
        "technology": "https://www.theguardian.com/technology",
        "science": "https://www.theguardian.com/science",
        "environment": "https://www.theguardian.com/environment",
    }

    try:
        driver = build_chrome_driver(headless=True)
    except Exception as exc:
        logger.warning("Failed to initialize Guardian Selenium scraper: %s", exc)
        return

    try:
        for category, section_url in sections.items():
            logger.info("Scraping Guardian section %s", category)
            try:
                driver.get(section_url)
                time.sleep(3)
                article_links = driver.find_elements(By.CSS_SELECTOR, "a[data-link-name='article']")
                if not article_links:
                    article_links = driver.find_elements(By.CSS_SELECTOR, "div.fc-item__container a")

                seen_urls: set[str] = set()
                for link_element in article_links[:10]:
                    article_url = link_element.get_attribute("href")
                    if not article_url or article_url in seen_urls or "theguardian.com" not in article_url:
                        continue
                    seen_urls.add(article_url)
                    text = fetch_and_extract(article_url)
                    if not text or len(text) < 100:
                        continue
                    title = (link_element.text or "").strip()
                    if not title:
                        title = article_url.split("/")[-1].replace("-", " ").title() or "The Guardian Article"
                    yield _article(title, article_url, text, "the-guardian")
            except Exception as exc:
                logger.warning("Guardian Selenium section %s failed: %s", category, exc)
    finally:
        driver.quit()


def scrape_france24_selenium_stream(
    *,
    check_link_pool: bool = True,
    track_links: bool = True,
) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    from selenium.webdriver.common.by import By

    sections = {
        "world": "https://www.france24.com/en/",
        "europe": "https://www.france24.com/en/europe/",
        "americas": "https://www.france24.com/en/americas/",
        "middle-east": "https://www.france24.com/en/middle-east/",
        "africa": "https://www.france24.com/en/africa/",
        "asia-pacific": "https://www.france24.com/en/asia-pacific/",
    }

    try:
        driver = build_chrome_driver(headless=True)
    except Exception as exc:
        logger.warning("Failed to initialize France24 Selenium scraper: %s", exc)
        return

    try:
        for category, section_url in sections.items():
            logger.info("Scraping France24 section %s", category)
            try:
                driver.get(section_url)
                time.sleep(3)
                article_links = driver.find_elements(By.CSS_SELECTOR, "article a.article__title-link")
                if not article_links:
                    article_links = driver.find_elements(By.CSS_SELECTOR, ".m-item-list-article a")
                if not article_links:
                    article_links = driver.find_elements(By.CSS_SELECTOR, "h2 a")

                seen_urls: set[str] = set()
                for link_element in article_links[:10]:
                    article_url = link_element.get_attribute("href")
                    if not article_url:
                        continue
                    article_url = urljoin("https://www.france24.com", article_url)
                    if article_url in seen_urls or "france24.com" not in article_url:
                        continue
                    seen_urls.add(article_url)
                    text = fetch_and_extract(article_url)
                    if not text or len(text) < 100:
                        continue
                    title = (link_element.text or "").strip()
                    if not title:
                        title = article_url.split("/")[-1].replace("-", " ").title() or "France24 Article"
                    if len(title) < 10:
                        continue
                    yield _article(title, article_url, text, "france24")
            except Exception as exc:
                logger.warning("France24 Selenium section %s failed: %s", category, exc)
    finally:
        driver.quit()


def scrape_npr_selenium_stream(
    *,
    check_link_pool: bool = True,
    track_links: bool = True,
) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    from selenium.webdriver.common.by import By

    sections = {
        "world": "https://www.npr.org/sections/world/",
        "business": "https://www.npr.org/sections/business/",
        "technology": "https://www.npr.org/sections/technology/",
    }

    try:
        driver = build_chrome_driver(headless=True)
    except Exception as exc:
        logger.warning("Failed to initialize NPR Selenium scraper: %s", exc)
        return

    try:
        for category, section_url in sections.items():
            logger.info("Scraping NPR section %s", category)
            try:
                driver.get(section_url)
                time.sleep(3)
                article_links = driver.find_elements(By.CSS_SELECTOR, "h2.title a")
                if not article_links:
                    article_links = driver.find_elements(By.CSS_SELECTOR, "article h3 a")

                seen_urls: set[str] = set()
                for link_element in article_links[:10]:
                    article_url = link_element.get_attribute("href")
                    if not article_url or article_url in seen_urls or "npr.org" not in article_url:
                        continue
                    seen_urls.add(article_url)
                    text = fetch_and_extract(article_url)
                    if not text or len(text) < 100:
                        continue
                    title = (link_element.text or "").strip()
                    if not title:
                        parts = article_url.rstrip("/").split("/")
                        title = parts[-2].replace("-", " ").title() if len(parts) >= 2 else "NPR Article"
                    if len(title) < 10:
                        continue
                    yield _article(title, article_url, text, "npr")
            except Exception as exc:
                logger.warning("NPR Selenium section %s failed: %s", category, exc)
    finally:
        driver.quit()


def scrape_newsapi_stream(
    language: str = "en",
    page_size: int = 50,
    *,
    check_link_pool: bool = True,
    track_links: bool = True,
) -> Iterable[Dict[str, Any]]:
    del check_link_pool, track_links
    if not NEWSAPI_CONFIG.api_key:
        logger.info("NEWSAPI_KEY not configured; skipping NewsAPI scraper")
        return

    topic_query = (
        "politics OR government OR science OR research OR "
        "technology OR innovation OR health OR medicine OR business OR finance OR "
        "crime OR justice OR climate OR environment OR education OR war OR conflict"
    )
    target_date = datetime.now(timezone.utc).date()
    base_url = "https://newsapi.org/v2/everything"

    for page in [1, 2]:
        try:
            response = requests.get(
                base_url,
                params={
                    "q": topic_query,
                    "language": language,
                    "from": target_date.isoformat(),
                    "to": target_date.isoformat(),
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "page": page,
                    "apiKey": NEWSAPI_CONFIG.api_key,
                },
                timeout=10,
                headers=DEFAULT_HEADERS,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("NewsAPI request for page %s failed: %s", page, exc)
            continue

        for article in data.get("articles", []):
            url = article.get("url")
            title = (article.get("title") or "").strip()
            if not url or not title:
                continue
            text = fetch_and_extract(url)
            if not text:
                text = (article.get("description") or article.get("content") or "").strip()
            if not text:
                continue
            yield _article(title, url, text, "newsapi")
