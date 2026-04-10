"""Repository for pipeline logs."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pymongo import ASCENDING, DESCENDING

from news_crawler.core.config import MONGO_CONFIG
from news_crawler.db.mongo_client import get_db

logger = logging.getLogger(__name__)


class PipelineLogsRepository:
    """Store structured pipeline logs in MongoDB."""

    INDEX_SPECS: ClassVar[List[Tuple[List[Tuple[str, int]], Dict[str, Any]]]] = [
        ([("ts", DESCENDING)], {"name": "ts_-1"}),
        ([("actor", ASCENDING)], {"name": "actor_1"}),
        ([("action", ASCENDING)], {"name": "action_1"}),
        ([("status", ASCENDING)], {"name": "status_1"}),
        ([("batch_id", ASCENDING)], {"name": "batch_id_1"}),
        ([("article_id", ASCENDING), ("action", ASCENDING)], {"name": "article_id_1_action_1"}),
    ]

    def __init__(self) -> None:
        self.collection = get_db()[MONGO_CONFIG.logs_collection]
        self.ensure_indexes()

    def ensure_indexes(self) -> List[str]:
        return [self.collection.create_index(keys, **options) for keys, options in self.INDEX_SPECS]

    def create_index(self, keys: List[Tuple[str, int]], **kwargs: Any) -> str:
        return self.collection.create_index(keys, **kwargs)

    def log_event(
        self,
        action: str,
        actor: str,
        article_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> str:
        log_entry = {
            "ts": datetime.now(timezone.utc),
            "action": action,
            "actor": actor,
            "article_id": article_id,
            "batch_id": batch_id,
            "status": status,
            "details": details or {},
            "error_message": error_message,
        }
        try:
            result = self.collection.insert_one(log_entry)
            return str(result.inserted_id)
        except Exception as exc:
            logger.error("Failed to insert log: %s", exc)
            return ""

    def get_logs(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        status: Optional[str] = None,
        article_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if actor:
            query["actor"] = actor
        if action:
            query["action"] = action
        if status:
            query["status"] = status
        if article_id:
            query["article_id"] = article_id
        if batch_id:
            query["batch_id"] = batch_id

        cursor = self.collection.find(query).sort("ts", -1).limit(limit)
        return list(cursor)
