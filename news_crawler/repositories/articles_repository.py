"""MongoDB repository for articles."""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection

from news_crawler.core.config import MONGO_CONFIG
from news_crawler.db.mongo_client import get_db


class ArticlesRepository:
    """Repository for article documents."""

    INDEX_SPECS: ClassVar[List[Tuple[List[Tuple[str, int]], Dict[str, Any]]]] = [
        ([("sample", ASCENDING)], {"name": "sample_1"}),
        ([("source", ASCENDING)], {"name": "source_1"}),
        ([("scraped_at", DESCENDING)], {"name": "scraped_at_-1"}),
        ([("source", ASCENDING), ("scraped_at", DESCENDING)], {"name": "source_1_scraped_at_-1"}),
    ]

    def __init__(self) -> None:
        self.collection: Collection = get_db()[MONGO_CONFIG.articles_collection]

    def insert_article(self, data: Dict[str, Any]) -> str:
        result = self.collection.insert_one(data)
        return str(result.inserted_id)

    def create_articles(self, data: Dict[str, Any]) -> str:
        return self.insert_article(data)

    def aggregate_articles(self, pipeline: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        return self.collection.aggregate(pipeline)

    def get_articles(
        self, params: Dict[str, Any], projection: Optional[Dict[str, int]] = None
    ) -> Iterable[Dict[str, Any]]:
        return self.collection.find(params, projection) if projection else self.collection.find(params)

    def get_one_article(
        self, params: Dict[str, Any], sorting: Optional[List[Tuple[str, int]]] = None
    ) -> Optional[Dict[str, Any]]:
        return self.collection.find_one(params, sort=sorting) if sorting else self.collection.find_one(params)

    def update_articles(self, selector: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        result = self.collection.update_one(selector, update_data)
        return result.modified_count

    def delete_articles(self, selector: Dict[str, Any]) -> int:
        result = self.collection.delete_many(selector)
        return result.deleted_count

    def count_articles(self, params: Dict[str, Any]) -> int:
        return self.collection.count_documents(params)

    def get_articles_grouped_by_source(self) -> Dict[str, List[Dict[str, Any]]]:
        pipeline = [{"$group": {"_id": "$source", "articles": {"$push": "$$ROOT"}}}]
        grouped_result = self.collection.aggregate(pipeline)
        return {group["_id"]: group["articles"] for group in grouped_result}

    def ensure_indexes(self) -> List[str]:
        return [self.collection.create_index(keys, **options) for keys, options in self.INDEX_SPECS]

    def create_index(self, keys: List[Tuple[str, int]], **kwargs) -> str:
        return self.collection.create_index(keys, **kwargs)
