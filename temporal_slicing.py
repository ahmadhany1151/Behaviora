"""
Member 2: Temporal Framing & Multi-Cutoff Slicing Engine (Phase 2)
Customer Intelligence Platform - Zero UI Sprint

Generates leakage-free 180-day feature windows and 30-day label windows
across Validation (END - 60d), Training (END - 30d), and Scoring (END) cutoffs.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import pandas as pd


def get_window_slices(
    df: pd.DataFrame,
    cutoff_date: Union[str, pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split transaction records strictly around cutoff_date without data leakage:
      - Feature Window: (cutoff_date - 180 days, cutoff_date]
      - Label Window:   (cutoff_date, cutoff_date + 30 days]
    """
    cutoff = pd.to_datetime(cutoff_date)
    feature_start = cutoff - pd.Timedelta(days=180)
    label_end = cutoff + pd.Timedelta(days=30)

    if not pd.api.types.is_datetime64_any_dtype(df["t_dat"]):
        df = df.copy()
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    feature_mask = (df["t_dat"] > feature_start) & (df["t_dat"] <= cutoff)
    label_mask = (df["t_dat"] > cutoff) & (df["t_dat"] <= label_end)

    feature_window_df = df.loc[feature_mask].copy()
    label_window_df = df.loc[label_mask].copy()

    return feature_window_df, label_window_df


def build_temporal_dataset(
    df: pd.DataFrame,
    cutoff_date: Union[str, pd.Timestamp],
    include_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build active (customer_id, category_id) pairs from the 180-day feature window
    and optionally attach binary repurchase labels from the 30-day label window.

    Returns:
        (feature_window_df, active_pairs_df)
    """
    feature_window_df, label_window_df = get_window_slices(df, cutoff_date)

    active_pairs_df = (
        feature_window_df[["customer_id", "category_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if include_labels:
        future_purchases = (
            label_window_df[["customer_id", "category_id"]]
            .drop_duplicates()
            .assign(label=1)
        )

        active_pairs_df = active_pairs_df.merge(
            future_purchases,
            on=["customer_id", "category_id"],
            how="left",
        )
        active_pairs_df["label"] = active_pairs_df["label"].fillna(0).astype(int)

    return feature_window_df, active_pairs_df


def generate_all_splits(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "data/splits",
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate and save Validation (END - 60d), Training (END - 30d),
    and Scoring (END, include_labels=False) datasets.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    end_date = pd.to_datetime(df["t_dat"]).max()
    splits_config = [
        ("val", end_date - pd.Timedelta(days=60), True),
        ("train", end_date - pd.Timedelta(days=30), True),
        ("score", end_date, False),
    ]

    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for split_name, cutoff, has_labels in splits_config:
        feat_df, pairs_df = build_temporal_dataset(
            df,
            cutoff_date=cutoff,
            include_labels=has_labels,
        )

        feat_df.to_parquet(out_path / f"{split_name}_feature_window.parquet", index=False)
        pairs_df.to_parquet(out_path / f"{split_name}_active_pairs.parquet", index=False)

        results[split_name] = (feat_df, pairs_df)

        summary = (
            f"[{split_name.upper()}] cutoff={cutoff.date()} | "
            f"feature_rows={len(feat_df):,} | active_pairs={len(pairs_df):,}"
        )
        if has_labels:
            pos_rate = pairs_df["label"].mean() * 100
            summary += f" | positive_rate={pos_rate:.2f}%"
        print(summary)

    return results


def _load_clean_transactions(
    parquet_path: Union[str, Path] = "clean_transactions.parquet",
    raw_csv_path: Union[str, Path] = "transactions_with_articles.csv",
) -> pd.DataFrame:
    """
    Load Member 1's clean_transactions.parquet if available; otherwise fallback
    to deterministic preprocessing from transactions_with_articles.csv.
    """
    p_path = Path(parquet_path)
    if p_path.exists():
        print(f"Loading {p_path}...")
        df = pd.read_parquet(p_path)
        df["t_dat"] = pd.to_datetime(df["t_dat"])
        return df

    c_path = Path(raw_csv_path)
    if not c_path.exists():
        raise FileNotFoundError(
            f"Neither '{p_path}' nor '{c_path}' was found in the working directory."
        )

    print(f"'{p_path}' not found. Building schema from '{c_path}'...")
    df = pd.read_csv(c_path, dtype={"customer_id": str, "article_id": str})
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    unique_cats = sorted(df["garment_group_name"].dropna().unique())
    cat_to_id = {cat: idx for idx, cat in enumerate(unique_cats)}
    df["category_id"] = df["garment_group_name"].map(cat_to_id).astype(int)

    return df


if __name__ == "__main__":
    transactions_df = _load_clean_transactions()
    splits = generate_all_splits(transactions_df)

    # Verify Member 2 checkpoint at END - 30d (2020-08-23)
    _, train_pairs = splits["train"]
    actual_pairs = len(train_pairs)
    actual_rate = train_pairs["label"].mean() * 100

    assert actual_pairs == 42099, f"Checkpoint failed: expected 42,099 pairs, got {actual_pairs}"
    assert abs(actual_rate - 9.01) < 0.05, f"Checkpoint failed: expected ~9.01%, got {actual_rate:.2f}%"

    print("
[PASS] Member 2 Checkpoint Verified: 42,099 active pairs | 9.01% positive rate.")
