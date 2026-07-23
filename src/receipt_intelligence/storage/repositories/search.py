"""Item retrieval using SQLite FTS5 with deterministic lexical fallback."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Any

from receipt_intelligence.storage.normalization import (
    CATEGORY_ALIASES,
    normalize_merchant_name,
    normalize_text,
    tokenize,
)
from receipt_intelligence.storage.repositories.base import BaseRepository, fts_available


class SearchRepository(BaseRepository):
    def search_items(
        self,
        *,
        semantic_query: str,
        merchant: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        categories: list[str] | None = None,
        limit: int | None = 25,
    ) -> list[dict[str, Any]]:
        merchant_normalized = normalize_merchant_name(merchant) if merchant else None
        query_text = semantic_query.strip()
        expanded_terms = self.expand_query_terms(query_text) if query_text else []
        fts_query = self._build_fts_query(expanded_terms)

        with self.connect() as connection:
            if fts_available(connection) and fts_query:
                results = self._search_items_fts(
                    connection,
                    fts_query,
                    merchant_normalized,
                    date_from,
                    date_to,
                    categories,
                    limit,
                )
                if results:
                    return results
            return self._search_items_fallback(
                connection,
                expanded_terms,
                merchant_normalized,
                date_from,
                date_to,
                categories,
                limit,
            )

    def _build_filter_sql(
        self,
        *,
        merchant_normalized: str | None,
        date_from: str | None,
        date_to: str | None,
        categories: list[str] | None,
        where_prefix: str = "WHERE",
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if merchant_normalized:
            clauses.append("r.merchant_normalized = ?")
            parameters.append(merchant_normalized)
        if date_from:
            clauses.append("r.receipt_date >= ?")
            parameters.append(date_from)
        if date_to:
            clauses.append("r.receipt_date <= ?")
            parameters.append(date_to)
        if categories:
            placeholders = ",".join("?" for _ in categories)
            clauses.append(f"i.category IN ({placeholders})")
            parameters.extend(categories)
        if not clauses:
            return "", parameters
        return f" {where_prefix} " + " AND ".join(clauses), parameters

    def _search_items_fts(
        self,
        connection: sqlite3.Connection,
        fts_query: str,
        merchant_normalized: str | None,
        date_from: str | None,
        date_to: str | None,
        categories: list[str] | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        filter_sql, parameters = self._build_filter_sql(
            merchant_normalized=merchant_normalized,
            date_from=date_from,
            date_to=date_to,
            categories=categories,
            where_prefix="AND",
        )
        sql = f"""
            SELECT
                i.id AS item_id,
                r.id AS receipt_db_id,
                r.job_id,
                r.merchant_name,
                r.merchant_normalized,
                r.receipt_date,
                r.receipt_time,
                r.currency,
                r.grand_total AS receipt_total,
                r.approved_receipt_path,
                r.image_path,
                i.raw_name,
                i.normalized_name,
                i.category,
                i.parser_item_type,
                i.category_group,
                i.category_key,
                i.quantity,
                i.unit,
                i.unit_price,
                i.original_price,
                i.discount_amount,
                i.line_total,
                i.tax_code,
                i.confidence,
                i.embedding_text,
                bm25(receipt_item_fts) AS score
            FROM receipt_item_fts
            JOIN receipt_items i ON i.id = receipt_item_fts.item_id
            JOIN receipts r ON r.id = i.receipt_id
            WHERE receipt_item_fts MATCH ?
            {filter_sql}
            ORDER BY score ASC
            {"LIMIT ?" if limit is not None else ""}
        """
        sql_parameters = [fts_query, *parameters]
        if limit is not None:
            sql_parameters.append(max(1, int(limit)))
        rows = connection.execute(sql, sql_parameters).fetchall()
        return [
            self._row_to_match(row, rank=index + 1, scoring="sqlite_fts5_bm25")
            for index, row in enumerate(rows)
        ]

    def _search_items_fallback(
        self,
        connection: sqlite3.Connection,
        expanded_terms: list[str],
        merchant_normalized: str | None,
        date_from: str | None,
        date_to: str | None,
        categories: list[str] | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        filter_sql, parameters = self._build_filter_sql(
            merchant_normalized=merchant_normalized,
            date_from=date_from,
            date_to=date_to,
            categories=categories,
            where_prefix="WHERE",
        )
        sql = f"""
            SELECT
                i.id AS item_id,
                r.id AS receipt_db_id,
                r.job_id,
                r.merchant_name,
                r.merchant_normalized,
                r.receipt_date,
                r.receipt_time,
                r.currency,
                r.grand_total AS receipt_total,
                r.approved_receipt_path,
                r.image_path,
                i.raw_name,
                i.normalized_name,
                i.category,
                i.parser_item_type,
                i.category_group,
                i.category_key,
                i.quantity,
                i.unit,
                i.unit_price,
                i.original_price,
                i.discount_amount,
                i.line_total,
                i.tax_code,
                i.confidence,
                i.embedding_text,
                0.0 AS score
            FROM receipt_items i
            JOIN receipts r ON r.id = i.receipt_id
            {filter_sql}
        """
        scored: list[tuple[float, sqlite3.Row]] = []
        query_terms = set(normalize_text(" ".join(expanded_terms)).split())
        for row in connection.execute(sql, parameters).fetchall():
            normalized_embedding = normalize_text(row["embedding_text"])
            text_terms = set(normalized_embedding.split())
            if not query_terms:
                score = 0.0
            else:
                hits = len(query_terms & text_terms)
                substring_hits = sum(
                    1 for term in query_terms if term and term in normalized_embedding
                )
                score = hits + 0.35 * substring_hits
            if score > 0 or not expanded_terms:
                scored.append((score, row))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["receipt_date"] or ""))
        selected = scored if limit is None else scored[: max(1, int(limit))]
        return [
            self._row_to_match(
                row,
                rank=index + 1,
                scoring="lexical_fallback",
                score=score,
            )
            for index, (score, row) in enumerate(selected)
        ]

    @staticmethod
    def _row_to_match(
        row: sqlite3.Row,
        *,
        rank: int,
        scoring: str,
        score: float | None = None,
    ) -> dict[str, Any]:
        value = dict(row)
        value["rank"] = rank
        value["scoring"] = scoring
        if score is not None:
            value["score"] = round(float(score), 4)
        elif value.get("score") is not None:
            try:
                value["score"] = round(float(value["score"]), 4)
            except Exception:
                pass
        value["evidence"] = {
            "matched_item": value.get("raw_name"),
            "normalized_name": value.get("normalized_name"),
            "product_category": value.get("category"),
            "parser_item_type": value.get("parser_item_type"),
            "category_group": value.get("category_group"),
            "category_key": value.get("category_key"),
            "merchant": value.get("merchant_name") or value.get("merchant_normalized"),
            "date": value.get("receipt_date"),
            "line_total": value.get("line_total"),
        }
        return value

    def expand_query_terms(self, query: str) -> list[str]:
        normalized_query = normalize_text(query)
        terms: list[str] = []
        seen: set[str] = set()

        def add(term: str) -> None:
            normalized = normalize_text(term)
            if normalized and normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)

        add(normalized_query)
        for token in tokenize(normalized_query):
            add(token)
        for category, aliases in CATEGORY_ALIASES.items():
            category_match = normalize_text(category).replace(" ", "") in normalized_query.replace(
                " ", ""
            )
            alias_match = any(normalize_text(alias) in normalized_query for alias in aliases)
            if alias_match or category_match:
                add(category)
                for alias in aliases:
                    add(alias)
        return terms[:80]

    def infer_categories(self, query: str) -> list[str]:
        normalized_query = normalize_text(query)
        hygiene_terms = {
            "hygiene",
            "hygieneartikel",
            "personal care",
            "koerperpflege",
            "körperpflege",
        }
        shampoo_terms = {
            "shampoo",
            "hair shampoo",
            "haar shampoo",
            "haarpflege",
            "head shoulders",
            "headandshoulders",
            "h s",
            "elvital",
            "fructis",
        }
        if any(normalize_text(term) in normalized_query for term in hygiene_terms):
            return ["personal_care/hygiene", "personal_care/shampoo"]
        if any(normalize_text(term) in normalized_query for term in shampoo_terms):
            return ["personal_care/shampoo"]

        categories: list[str] = []
        for category, aliases in CATEGORY_ALIASES.items():
            if normalize_text(category) in normalized_query:
                categories.append(category)
                continue
            if any(normalize_text(alias) in normalized_query for alias in aliases):
                categories.append(category)
        return categories

    @staticmethod
    def _build_fts_query(terms: Iterable[str]) -> str:
        phrases: list[str] = []
        for term in terms:
            tokens = [token for token in tokenize(term) if len(token) >= 2]
            if not tokens:
                continue
            if len(tokens) == 1:
                phrases.append(f'"{tokens[0]}"')
            else:
                phrases.append('"' + " ".join(tokens[:6]) + '"')
            if len(phrases) >= 20:
                break
        return " OR ".join(phrases)
