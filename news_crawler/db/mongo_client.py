"""MongoDB client factory for the news crawler."""
from __future__ import annotations

import logging
from typing import Optional

from pymongo import MongoClient

from news_crawler.core.config import MONGO_CONFIG

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_db = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        if not MONGO_CONFIG.uri:
            raise RuntimeError("MONGO_URI is required to create a MongoDB client")
        _client = MongoClient(
            MONGO_CONFIG.uri,
            appname="news-crawler-ai",
            serverSelectionTimeoutMS=8000,
        )
    return _client


def get_db():
    global _db
    if _db is None:
        if not MONGO_CONFIG.db_name:
            raise RuntimeError("MONGODB_DB is required to select a database")
        _db = get_client()[MONGO_CONFIG.db_name]
    return _db
