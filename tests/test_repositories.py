"""Tests for Mongo repository wrappers using fake collections."""
from __future__ import annotations

from types import SimpleNamespace

from news_crawler.repositories.articles_repository import ArticlesRepository
from news_crawler.repositories.link_pool_repository import LinkPoolRepository
from news_crawler.repositories.metadata_repository import MetadataRepository
from news_crawler.repositories.pipeline_logs_repository import PipelineLogsRepository


class _Cursor(list):
    def sort(self, *args, **kwargs):
        self.sort_args = (args, kwargs)
        return self

    def limit(self, limit):
        self.limit_value = limit
        return self


class _Collection:
    def __init__(self) -> None:
        self.inserted: list[dict] = []
        self.indexes: list[tuple] = []
        self.find_one_result = None
        self.find_result = _Cursor()
        self.aggregate_result = []
        self.modified_count = 1
        self.deleted_count = 2

    def insert_one(self, data):
        self.inserted.append(data)
        return SimpleNamespace(inserted_id="inserted-id")

    def aggregate(self, pipeline):
        self.aggregate_pipeline = pipeline
        return self.aggregate_result

    def find(self, *args, **kwargs):
        self.find_args = (args, kwargs)
        return self.find_result

    def find_one(self, *args, **kwargs):
        self.find_one_args = (args, kwargs)
        return self.find_one_result

    def update_one(self, *args, **kwargs):
        self.update_args = (args, kwargs)
        return SimpleNamespace(modified_count=self.modified_count)

    def delete_many(self, selector):
        self.deleted_selector = selector
        return SimpleNamespace(deleted_count=self.deleted_count)

    def count_documents(self, params):
        self.count_params = params
        return 7

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))
        return kwargs.get("name", "index-name")

    def find_one_and_update(self, *args, **kwargs):
        self.find_one_and_update_args = (args, kwargs)
        return {"url": args[0]["url"], "is_articles_processed": False}


def test_articles_repository_methods_use_collection():
    collection = _Collection()
    repo = object.__new__(ArticlesRepository)
    repo.collection = collection
    collection.aggregate_result = [{"_id": "bbc", "articles": [{"title": "A"}]}]

    assert repo.insert_article({"title": "A"}) == "inserted-id"
    assert repo.create_articles({"title": "B"}) == "inserted-id"
    assert list(repo.aggregate_articles([{"$match": {}}])) == collection.aggregate_result
    assert repo.count_articles({"source": "bbc"}) == 7
    assert repo.update_articles({"id": 1}, {"$set": {"x": 1}}) == 1
    assert repo.delete_articles({"source": "old"}) == 2
    assert repo.get_articles_grouped_by_source() == {"bbc": [{"title": "A"}]}
    assert repo.ensure_indexes()


def test_link_pool_repository_deduplication_methods():
    collection = _Collection()
    repo = object.__new__(LinkPoolRepository)
    repo.collection = collection

    assert repo.insert_link({"url": "https://example.com"}) == "inserted-id"
    assert repo.upsert_link("https://example.com") == {
        "url": "https://example.com",
        "is_articles_processed": False,
    }
    collection.find_one_result = {"is_articles_processed": True}
    assert repo.is_processed("https://example.com") is True
    assert repo.mark_as_processed("https://example.com", "batch-1") == 1
    assert repo.ensure_indexes()


def test_metadata_repository_methods_use_collection():
    collection = _Collection()
    repo = object.__new__(MetadataRepository)
    repo.collection = collection

    assert repo.insert_metadata("batch-1", {"articles_processed": 3}) == "inserted-id"
    assert collection.inserted[-1] == {"batch_id": "batch-1", "articles_processed": 3}
    assert repo.get_metadata({"batch_id": "batch-1"}) == collection.find_result
    assert repo.get_one_metadata({"batch_id": "batch-1"}) is None
    assert repo.update_metadata({"batch_id": "batch-1"}, {"$set": {"done": True}}).modified_count == 1
    assert repo.delete_metadata_many({"old": True}) == 2
    assert repo.ensure_indexes()


def test_pipeline_logs_repository_logs_and_filters():
    collection = _Collection()
    repo = object.__new__(PipelineLogsRepository)
    repo.collection = collection

    assert repo.log_event(
        action="process_article",
        actor="orchestrator",
        article_id="article-1",
        batch_id="batch-1",
        status="success",
        details={"step": "save"},
    ) == "inserted-id"
    assert collection.inserted[-1]["action"] == "process_article"
    assert collection.inserted[-1]["details"] == {"step": "save"}

    collection.find_result = _Cursor([{"action": "process_article"}])
    logs = repo.get_logs(actor="orchestrator", status="success", limit=10)
    assert logs == [{"action": "process_article"}]
    assert collection.find_args[0][0] == {"actor": "orchestrator", "status": "success"}

