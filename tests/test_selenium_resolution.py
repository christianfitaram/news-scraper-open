"""Tests for Selenium driver and browser path resolution helpers."""
from __future__ import annotations

from pathlib import Path

from news_crawler.scrapers import streams


def _force_no_system_chromedriver(monkeypatch) -> None:
    real_exists = streams.os.path.exists

    def fake_exists(path: str) -> bool:
        if path in {"/usr/local/bin/chromedriver", "/usr/bin/chromedriver"}:
            return False
        return real_exists(path)

    monkeypatch.setattr(streams.os.path, "exists", fake_exists)


def test_resolve_optional_chromedriver_path_ignores_missing_env(monkeypatch, tmp_path: Path) -> None:
    driver_path = tmp_path / "chromedriver"
    driver_path.write_text("driver", encoding="utf-8")

    monkeypatch.setenv("CHROMEDRIVER_PATH", str(tmp_path / "missing-driver"))
    monkeypatch.setattr(
        streams,
        "_install_chromedriver_with_manager",
        lambda: (_ for _ in ()).throw(AssertionError("manager should not be called")),
    )
    monkeypatch.setattr(
        streams.shutil,
        "which",
        lambda name: str(driver_path) if name == "chromedriver" else None,
    )
    _force_no_system_chromedriver(monkeypatch)

    assert streams._resolve_optional_chromedriver_path() == str(driver_path)


def test_resolve_optional_chromedriver_path_falls_back_to_manager(monkeypatch, tmp_path: Path) -> None:
    managed_driver = tmp_path / "managed-chromedriver"
    managed_driver.write_text("driver", encoding="utf-8")

    monkeypatch.delenv("CHROMEDRIVER_PATH", raising=False)
    monkeypatch.setattr(streams.shutil, "which", lambda _: None)
    monkeypatch.setattr(streams, "_install_chromedriver_with_manager", lambda: str(managed_driver))
    _force_no_system_chromedriver(monkeypatch)

    assert streams._resolve_optional_chromedriver_path() == str(managed_driver)


def test_resolve_optional_chrome_binary_prefers_valid_fallback_env(monkeypatch, tmp_path: Path) -> None:
    chrome_path = tmp_path / "Google Chrome"
    chrome_path.write_text("chrome", encoding="utf-8")

    monkeypatch.setenv("CHROME_BIN", str(tmp_path / "missing-chrome"))
    monkeypatch.setenv("CHROME_BINARY", str(chrome_path))
    monkeypatch.setattr(streams.shutil, "which", lambda _: None)

    assert streams._resolve_optional_chrome_binary() == str(chrome_path)
