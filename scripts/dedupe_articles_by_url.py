#!/usr/bin/env python3
"""Deduplicate articles by exact URL."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from news_crawler.core.config import validate_config
from news_crawler.repositories.articles_repository import ArticlesRepository


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deduplicate articles collection by exact URL.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete duplicate articles. Without this flag the script only reports what it would remove.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only inspect/process the first N duplicate URL groups.",
    )
    return parser


def _duplicate_groups(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    pipeline: List[Dict[str, Any]] = [
        {"$match": {"url": {"$exists": True, "$type": "string", "$ne": ""}}},
        {"$group": {"_id": "$url", "ids": {"$push": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
        {"$sort": {"count": -1, "_id": 1}},
    ]
    if limit is not None:
        pipeline.append({"$limit": limit})
    return list(ArticlesRepository().aggregate_articles(pipeline))


def _parse_scraped_at(value: Any) -> float:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return 0.0
    else:
        return 0.0

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _entity_count(document: Dict[str, Any]) -> int:
    return sum(len(document.get(field) or []) for field in ("locations", "organizations", "persons"))


def _document_score(document: Dict[str, Any]) -> tuple[int, int, int, int, int, float, str]:
    text_length = len((document.get("text") or "").strip())
    summary_length = len((document.get("summary") or "").strip())
    entity_count = _entity_count(document)
    has_topic = 1 if document.get("topic") else 0
    has_sentiment = 1 if (document.get("sentiment") or {}).get("label") else 0
    scraped_at_score = _parse_scraped_at(document.get("scraped_at"))
    return (
        text_length,
        summary_length,
        entity_count,
        has_topic,
        has_sentiment,
        scraped_at_score,
        str(document.get("_id", "")),
    )


def _pick_keeper(documents: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return max(documents, key=_document_score)


def _process_duplicate_group(repository: ArticlesRepository, url: str, ids: List[Any], *, apply: bool) -> Dict[str, Any]:
    documents = list(repository.get_articles({"_id": {"$in": ids}}))
    keeper = _pick_keeper(documents)
    remove_ids = [doc["_id"] for doc in documents if doc["_id"] != keeper["_id"]]

    deleted_count = 0
    if apply and remove_ids:
        deleted_count = repository.delete_articles({"_id": {"$in": remove_ids}})

    return {
        "url": url,
        "count": len(documents),
        "keep_id": keeper["_id"],
        "delete_ids": remove_ids,
        "deleted_count": deleted_count,
    }


def _print_summary(results: List[Dict[str, Any]], *, apply: bool) -> None:
    total_groups = len(results)
    total_duplicates = sum(len(item["delete_ids"]) for item in results)
    total_deleted = sum(item["deleted_count"] for item in results)

    print(f"Duplicate URL groups: {total_groups}")
    print(f"Duplicate article docs: {total_duplicates}")
    if apply:
        print(f"Deleted article docs: {total_deleted}")
    else:
        print("Dry run only. No documents were deleted.")

    preview = results[:10]
    if preview:
        print("\nPreview:")
        for item in preview:
            print(
                f"- url={item['url']} total={item['count']} keep={item['keep_id']} "
                f"remove={len(item['delete_ids'])}"
            )


def main() -> None:
    args = _build_parser().parse_args()
    validate_config(require_db=True)

    repository = ArticlesRepository()
    groups = _duplicate_groups(limit=args.limit)
    results = [
        _process_duplicate_group(repository, item["_id"], list(item["ids"]), apply=args.apply)
        for item in groups
    ]
    _print_summary(results, apply=args.apply)


if __name__ == "__main__":
    main()
