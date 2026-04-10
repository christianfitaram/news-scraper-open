"""Repository for batch metadata."""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection

from news_crawler.core.config import MONGO_CONFIG
from news_crawler.db.mongo_client import get_db


class MetadataRepository:
    """Repository handling pipeline metadata."""

    INDEX_SPECS: ClassVar[List[Tuple[List[Tuple[str, int]], Dict[str, Any]]]] = [
        ([("batch_id", ASCENDING)], {"name": "batch_id_1", "unique": True}),
        ([("started_at", DESCENDING)], {"name": "started_at_-1"}),
        ([("finished_at", DESCENDING)], {"name": "finished_at_-1"}),
    ]

    def __init__(self) -> None:
        self.collection: Collection = get_db()[MONGO_CONFIG.metadata_collection]

    def insert_metadata(self, batch_id: str, data: Dict[str, Any]) -> str:
        payload = {"batch_id": batch_id, **data}
        result = self.collection.insert_one(payload)
        return str(result.inserted_id)

    def get_metadata(
        self, param: Dict[str, Any], sorting: Optional[List[Tuple[str, int]]] = None
    ):
        return self.collection.find(param, sort=sorting) if sorting else self.collection.find(param)

    def get_one_metadata(
        self, param: Dict[str, Any], sorting: Optional[List[Tuple[str, int]]] = None
    ):
        return self.collection.find_one(param, sort=sorting) if sorting else self.collection.find_one(param)

    def update_metadata(self, selector: Dict[str, Any], update_data: Dict[str, Any]):
        return self.collection.update_one(selector, update_data)

    def delete_metadata_many(self, selector: Dict[str, Any]) -> int:
        result = self.collection.delete_many(selector)
        return result.deleted_count

    def ensure_indexes(self) -> List[str]:
        return [self.collection.create_index(keys, **options) for keys, options in self.INDEX_SPECS]

    def create_index(self, keys: List[Tuple[str, int]], **kwargs) -> str:
        return self.collection.create_index(keys, **kwargs)
