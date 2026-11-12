from __future__ import annotations

import pandas as pd


def merge_feature_tables(feature_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not feature_tables:
        raise ValueError("feature_tables cannot be empty")

    merged_parts = []
    ids = None
    labels = None
    for name, table in feature_tables.items():
        if ids is None:
            ids = table["id"].tolist()
            labels = table["label"].tolist()
        elif ids != table["id"].tolist() or labels != table["label"].tolist():
            raise ValueError(f"sample order mismatch in feature table: {name}")
        merged_parts.append(table.drop(columns=["id", "label"]).reset_index(drop=True))

    return pd.concat(
        [pd.DataFrame({"id": ids, "label": labels}), *merged_parts],
        axis=1,
        verify_integrity=True,
    )

