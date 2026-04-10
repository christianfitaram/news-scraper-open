"""Repository for link_pool collection."""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, List, Tuple

from pymongo import ASCENDING, ReturnDocument
from pymongo.collection import Collection

from news_crawler.core.config import MONGO_CONFIG
from news_crawler.db.mongo_client import get_db


class LinkPoolRepository:
    """Repository handling URL deduplication and processing state."""

    INDEX_SPECS: ClassVar[List[Tuple[List[Tuple[str, int]], Dict[str, Any]]]] = [
        ([("url", ASCENDING)], {"name": "url_1", "unique": True}),
        ([("is_articles_processed", ASCENDING)], {"name": "is_articles_processed_1"}),
        ([("in_sample", ASCENDING)], {"name": "in_sample_1"}),
    ]

    def __init__(self) -> None:
        self.collection: Collection = get_db()[MONGO_CONFIG.link_pool_collection]

    def insert_link(self, data: Dict[str, Any]) -> str:
        result = self.collection.insert_one(data)
        return str(result.inserted_id)

    def update_link_in_pool(
        self,
        selector: Dict[str, Any],
        update_data: Dict[str, Any],
        *,
        upsert: bool = False,
    ) -> int:
        result = self.collection.update_one(selector, update_data, upsert=upsert)
        return result.modified_count

    def upsert_link(self, url: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        extra = extra or {}
        return self.collection.find_one_and_update(
            {"url": url},
            {"$setOnInsert": {"url": url}, "$set": extra},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def find_one_by_url(self, url: str, projection: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        return self.collection.find_one({"url": url}, projection=projection)

    def is_processed(self, url: str) -> bool:
        doc = self.collection.find_one({"url": url}, projection={"is_articles_processed": 1, "in_sample": 1})
        return bool(doc and (doc.get("is_articles_processed") or doc.get("in_sample")))

    def mark_as_processed(self, url: str, batch_id: str) -> int:
        res = self.collection.update_one(
            {"url": url},
            {"$set": {"is_articles_processed": True, "in_sample": batch_id}},
            upsert=True,
        )
        return res.modified_count

    def ensure_indexes(self) -> List[str]:
        return [self.collection.create_index(keys, **options) for keys, options in self.INDEX_SPECS]

    def create_index(self, keys: List[Tuple[str, int]], **kwargs) -> str:
        return self.collection.create_index(keys, **kwargs)
