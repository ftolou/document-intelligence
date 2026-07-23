"""Reference-data repository for product aliases."""

from __future__ import annotations

from receipt_intelligence.storage.normalization import CATEGORY_ALIASES, normalize_text, utc_now
from receipt_intelligence.storage.repositories.base import BaseRepository


class CatalogRepository(BaseRepository):
    def seed_product_aliases(self) -> None:
        now = utc_now()
        with self.connect() as connection:
            for category, aliases in CATEGORY_ALIASES.items():
                for alias in aliases:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO product_aliases(
                            alias, normalized_name, category, created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (normalize_text(alias), alias, category, now),
                    )
            connection.commit()
