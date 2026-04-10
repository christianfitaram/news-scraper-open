#!/usr/bin/env python3
"""Create MongoDB indexes for all news crawler repositories."""
from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Iterable, List, Protocol, Tuple

from news_crawler.core.config import MONGO_CONFIG, validate_config
from news_crawler.repositories.articles_repository import ArticlesRepository
from news_crawler.repositories.link_pool_repository import LinkPoolRepository
from news_crawler.repositories.metadata_repository import MetadataRepository
from news_crawler.repositories.pipeline_logs_repository import PipelineLogsRepository


IndexKeys = List[Tuple[str, int]]
IndexSpec = Tuple[IndexKeys, Dict[str, Any]]


class SupportsIndexBootstrap(Protocol):
    INDEX_SPECS: List[IndexSpec]
    collection: Any

    def create_index(self, keys: IndexKeys, **kwargs: Any) -> str:
        ...


RepositoryFactory = Callable[[], SupportsIndexBootstrap]


def _bootstrap_plan() -> List[Tuple[str, RepositoryFactory]]:
    return [
        (MONGO_CONFIG.articles_collection, ArticlesRepository),
        (MONGO_CONFIG.link_pool_collection, LinkPoolRepository),
        (MONGO_CONFIG.metadata_collection, MetadataRepository),
        (MONGO_CONFIG.logs_collection, PipelineLogsRepository),
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create MongoDB indexes for all crawler collections.")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate documents that block unique index creation before creating indexes.",
    )
    return parser


def _match_existing_fields(fields: IndexKeys) -> Dict[str, Dict[str, Any]]:
    return {field: {"$exists": True, "$ne": None} for field, _ in fields}


def _find_duplicate_groups(
    repository: SupportsIndexBootstrap,
    keys: IndexKeys,
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    group_id = {field: f"${field}" for field, _ in keys}
    pipeline = [
        {"$match": _match_existing_fields(keys)},
        {"$group": {"_id": group_id, "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
    ]
    return list(repository.collection.aggregate(pipeline))


def _document_score(document: Dict[str, Any]) -> Tuple[int, int, int, str]:
    non_empty_fields = sum(1 for value in document.values() if value not in (None, "", [], {}, False))
    return (
        1 if document.get("is_articles_processed") else 0,
        1 if document.get("in_sample") else 0,
        non_empty_fields,
        str(document.get("_id", "")),
    )


def _dedupe_unique_index(repository: SupportsIndexBootstrap, keys: IndexKeys) -> int:
    deleted_total = 0
    group_id = {field: f"${field}" for field, _ in keys}
    pipeline = [
        {"$match": _match_existing_fields(keys)},
        {"$group": {"_id": group_id, "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
    ]

    for group in repository.collection.aggregate(pipeline):
        ids = list(group["ids"])
        documents = list(repository.collection.find({"_id": {"$in": ids}}))
        keeper = max(documents, key=_document_score)
        to_delete = [doc["_id"] for doc in documents if doc["_id"] != keeper["_id"]]
        if to_delete:
            result = repository.collection.delete_many({"_id": {"$in": to_delete}})
            deleted_total += int(result.deleted_count)
    return deleted_total


def _ensure_repository_indexes(
    collection_name: str,
    repository: SupportsIndexBootstrap,
    *,
    dedupe: bool,
) -> List[str]:
    for keys, options in repository.INDEX_SPECS:
        if not options.get("unique"):
            continue

        duplicates = _find_duplicate_groups(repository, keys)
        if not duplicates:
            continue

        field_names = ", ".join(field for field, _ in keys)
        if not dedupe:
            examples = "\n".join(f"  - value={item['_id']} count={item['count']}" for item in duplicates)
            raise RuntimeError(
                f"Collection '{collection_name}' contains duplicates for unique index "
                f"'{options.get('name', field_names)}' on fields [{field_names}].\n"
                f"Examples:\n{examples}\n"
                f"Re-run with '--dedupe' to remove duplicates automatically."
            )

        deleted = _dedupe_unique_index(repository, keys)
        print(
            f"{collection_name}: removed {deleted} duplicate document(s) before creating "
            f"unique index '{options.get('name', field_names)}'"
        )

    return [repository.create_index(keys, **options) for keys, options in repository.INDEX_SPECS]


def _run_bootstrap(*, dedupe: bool) -> List[Tuple[str, List[str]]]:
    created_indexes: List[Tuple[str, List[str]]] = []
    for collection_name, factory in _bootstrap_plan():
        try:
            repository = factory()
            names = _ensure_repository_indexes(collection_name, repository, dedupe=dedupe)
        except Exception as exc:
            raise RuntimeError(f"Failed to bootstrap indexes for collection '{collection_name}': {exc}") from exc
        created_indexes.append((collection_name, names))
    return created_indexes


def _print_summary(results: Iterable[Tuple[str, List[str]]]) -> None:
    for collection_name, names in results:
        print(f"{collection_name}:")
        for name in names:
            print(f"  - {name}")


def main() -> None:
    args = _build_parser().parse_args()
    validate_config(require_db=True)
    results = _run_bootstrap(dedupe=args.dedupe)
    _print_summary(results)


if __name__ == "__main__":
    main()
