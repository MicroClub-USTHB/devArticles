"""
╔══════════════════════════════════════════════════════════════════════╗
║          structured_preprocessing.py — Tabular Data Pipeline        ║
║                                                                      ║
║  Complete preprocessing pipeline for structured/tabular data.        ║
║  Covers the full data cleaning and feature engineering lifecycle.    ║
║                                                                      ║
║  Pipeline steps (typical order):                                     ║
║    1. Inspect     — profile data, identify issues                    ║
║    2. Clean       — duplicates, outliers, type casting               ║
║    3. Impute      — fill missing values                              ║
║    4. Encode      — categorical → numerical                          ║
║    5. Scale       — normalize numeric features                       ║
║    6. Engineer    — create new features, time features               ║
║    7. Select      — remove low-info or redundant features            ║
║    8. Split       — train/val/test                                   ║
║                                                                      ║
║  Dependencies:                                                       ║
║    pip install pandas numpy scikit-learn scipy                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder, OneHotEncoder,
)
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger("structured_preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

#  BEGINNER — Loading and inspection
def load_data(
    path: Union[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Load tabular data from common formats with auto-detection.

    Supports: .csv, .tsv, .json, .jsonl, .parquet, .xlsx, .xls

    Extra kwargs are passed to the underlying pandas reader.

    Example:
        df = load_data("data.csv")
        df = load_data("data.xlsx", sheet_name="Sheet2")
        df = load_data("data.parquet")
    """
    path = Path(path)
    ext = path.suffix.lower()
    readers = {
        ".csv":     lambda: pd.read_csv(path, **kwargs),
        ".tsv":     lambda: pd.read_csv(path, sep="\t", **kwargs),
        ".json":    lambda: pd.read_json(path, **kwargs),
        ".jsonl":   lambda: pd.read_json(path, lines=True, **kwargs),
        ".parquet": lambda: pd.read_parquet(path, **kwargs),
        ".xlsx":    lambda: pd.read_excel(path, **kwargs),
        ".xls":     lambda: pd.read_excel(path, **kwargs),
    }
    if ext not in readers:
        raise ValueError(f"Unsupported format: {ext}")
    df = readers[ext]()
    logger.info(f"Loaded {path.name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def profile(df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a comprehensive column-by-column data profile.

    For each column, reports: dtype, n_missing, missing_pct, n_unique,
    min, max, mean, std, and sample values.

    Call this FIRST — before any transformation — to understand your data.

    Args:
        df:     Input DataFrame.
        target: Optional target column name (highlighted in output).

    Returns:
        Profile DataFrame (print it or inspect interactively).

    Example:
        profile_df = profile(df, target="price")
        print(profile_df.to_string())
    """
    rows = []
    for col in df.columns:
        s = df[col]
        n_missing = s.isnull().sum()
        row = {
            "column":       col,
            "dtype":        str(s.dtype),
            "n_missing":    n_missing,
            "missing_pct":  round(n_missing / len(df) * 100, 2),
            "n_unique":     s.nunique(),
            "is_target":    col == target,
        }
        if pd.api.types.is_numeric_dtype(s):
            row.update({
                "min":   round(s.min(), 4) if not s.isnull().all() else None,
                "max":   round(s.max(), 4) if not s.isnull().all() else None,
                "mean":  round(s.mean(), 4) if not s.isnull().all() else None,
                "std":   round(s.std(), 4) if not s.isnull().all() else None,
            })
        else:
            top = s.mode()[0] if not s.isnull().all() else None
            row.update({
                "min":  None, "max": None, "mean": None, "std": None,
                "top_value": top,
            })
        rows.append(row)

    result = pd.DataFrame(rows).set_index("column")
    logger.info(f"Profile: {df.shape[0]:,} rows, {df.shape[1]} columns, "
                f"{df.isnull().sum().sum():,} total missing values")
    return result

#  BEGINNER — Cleaning
def remove_duplicates(df: pd.DataFrame, subset: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Args:
        df:     DataFrame.
        subset: Columns to check. None = all columns.

    Returns:
        Deduplicated DataFrame.

    Example:
        df = remove_duplicates(df)
        df = remove_duplicates(df, subset=["email"])  # deduplicate by email
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    logger.info(f"Removed {removed:,} duplicate rows ({removed/before:.1%})")
    return df


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns where the fraction of missing values exceeds threshold.

    Args:
        df:        DataFrame.
        threshold: Drop if missing > threshold (default 0.5 = 50%).

    Example:
        # Drop columns with > 40% missing values
        df = drop_high_missing(df, threshold=0.4)
    """
    missing_rate = df.isnull().mean()
    to_drop = missing_rate[missing_rate > threshold].index.tolist()
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} high-missing columns: {to_drop}")
    return df.drop(columns=to_drop)


def cast_dtypes(df: pd.DataFrame, dtype_map: dict[str, str]) -> pd.DataFrame:
    """
    Cast columns to specified dtypes.

    Common dtype strings:
        "int32", "int64", "float32", "float64",
        "bool", "category", "datetime64[ns]", "string"

    Args:
        df:        DataFrame.
        dtype_map: {column_name: dtype_string}

    Example:
        df = cast_dtypes(df, {
            "age":       "int32",
            "price":     "float32",
            "category":  "category",
            "signup_dt": "datetime64[ns]",
        })
    """
    for col, dtype in dtype_map.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping dtype cast.")
            continue
        try:
            if dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            logger.warning(f"Could not cast '{col}' to {dtype}: {e}")
    return df


def clip_outliers(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Clip outliers in numeric columns using IQR or z-score.

    Methods:
        "iqr":    Clip values outside [Q1 - factor×IQR, Q3 + factor×IQR].
                  Robust to heavy-tailed distributions. factor=1.5 = Tukey's rule.
        "zscore": Clip values with |z| > factor.
                  Assumes roughly normal distribution. factor=3 is standard.

    Args:
        df:      DataFrame.
        columns: Columns to check. None = all numeric.
        method:  "iqr" | "zscore"
        factor:  Multiplier for the outlier threshold.

    Returns:
        DataFrame with clipped values (originals unchanged above/below bounds).

    Example:
        df = clip_outliers(df, method="iqr", factor=2.0)
        df = clip_outliers(df, columns=["salary"], method="zscore", factor=3.0)
    """
    df = df.copy()
    num_cols = columns or df.select_dtypes(include=np.number).columns.tolist()

    for col in num_cols:
        s = df[col].dropna()
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
        elif method == "zscore":
            lo = s.mean() - factor * s.std()
            hi = s.mean() + factor * s.std()
        else:
            raise ValueError(f"Unknown method: {method}")
        clipped = (df[col] < lo) | (df[col] > hi)
        if clipped.sum() > 0:
            logger.debug(f"Clipped {clipped.sum()} outliers in '{col}'")
        df[col] = df[col].clip(lower=lo, upper=hi)

    return df

# INTERMEDIATE — Imputation
def impute(
    df: pd.DataFrame,
    strategy: Union[str, dict] = "median",
    knn_neighbors: int = 5,
    constant_value: Any = 0,
) -> pd.DataFrame:
    """
    Fill missing values using one of several strategies.

    Strategies:
        "mean":     Replace NaN with column mean. Only for normal dists.
        "median":   Replace NaN with median. Robust to outliers. (recommended)
        "mode":     Most frequent value. Works for categoricals.
        "constant": Fill all NaNs with a single value.
        "knn":      K-nearest neighbors — uses similar rows' values.
                    Best quality, most expensive (O(n²)).
        "ffill":    Forward fill — use previous value (time series).
        "bfill":    Backward fill — use next value (time series).

    You can pass a dict to apply different strategies per column:
        strategy = {
            "age":        "median",
            "department": "mode",
            "notes":      "constant",
        }

    Args:
        df:              DataFrame.
        strategy:        String (applies to all columns) or dict per column.
        knn_neighbors:   Neighbors for KNN imputation.
        constant_value:  Fill value when strategy="constant".

    Returns:
        Imputed DataFrame.

    Example:
        df = impute(df, strategy="median")
        df = impute(df, strategy={"salary": "median", "dept": "mode"})
    """
    df = df.copy()

    # Build per-column strategy map
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if isinstance(strategy, str):
        strategy_map = {col: strategy for col in df.columns}
    else:
        strategy_map = strategy

    # Group by strategy for efficient processing
    from collections import defaultdict
    groups = defaultdict(list)
    for col in df.columns:
        s = strategy_map.get(col, "median" if col in num_cols else "mode")
        groups[s].append(col)

    for strat, cols in groups.items():
        cols = [c for c in cols if df[c].isnull().any()]
        if not cols:
            continue

        if strat in ("mean", "median", "most_frequent", "mode"):
            sk_strategy = "most_frequent" if strat == "mode" else strat
            num = [c for c in cols if c in num_cols]
            cat = [c for c in cols if c in cat_cols]
            if num:
                imp = SimpleImputer(strategy=sk_strategy if sk_strategy != "most_frequent" else "median")
                df[num] = imp.fit_transform(df[num])
            if cat:
                imp = SimpleImputer(strategy="most_frequent")
                df[cat] = imp.fit_transform(df[cat])

        elif strat == "constant":
            df[cols] = df[cols].fillna(constant_value)

        elif strat == "knn":
            num = [c for c in cols if c in num_cols]
            if num:
                imp = KNNImputer(n_neighbors=knn_neighbors)
                df[num] = imp.fit_transform(df[num])

        elif strat in ("ffill", "bfill"):
            df[cols] = df[cols].fillna(method=strat)

        logger.info(f"Imputed {cols} with strategy='{strat}'")

    return df

# INTERMEDIATE — Encoding
def encode_labels(
    df: pd.DataFrame,
    column: str,
    mapping: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Encode a target column (y) to integers.

    Args:
        df:      DataFrame.
        column:  Column to encode.
        mapping: Custom mapping {label: integer}. Auto-generated if None.

    Returns:
        (df_with_encoded_col, mapping_dict)

    Example:
        df, mapping = encode_labels(df, "category")
        # mapping = {"cat": 0, "dog": 1, "fish": 2}
        # Decode with: df["category"].map({v: k for k, v in mapping.items()})
    """
    df = df.copy()
    if mapping is None:
        unique_vals = sorted(df[column].dropna().unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
    df[column] = df[column].map(mapping)
    logger.info(f"Label encoded '{column}': {mapping}")
    return df, mapping


def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str],
    drop_first: bool = True,
    max_categories: int = 20,
) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns.

    One-hot encoding creates a binary column for each category.
    Best for unordered categoricals with low cardinality.

    WARNING: Avoid for high-cardinality columns (> 20 categories)
    — use target encoding or embeddings instead.

    Args:
        df:              DataFrame.
        columns:         Columns to encode.
        drop_first:      Drop the first dummy to avoid multicollinearity.
                         Keep True for linear models. False for trees.
        max_categories:  Skip encoding if column has too many categories.

    Returns:
        DataFrame with original cols replaced by dummies.

    Example:
        df = one_hot_encode(df, ["color", "size"])
        # "color" column (red/green/blue) → color_green, color_red
    """
    df = df.copy()
    to_encode = []
    for col in columns:
        n_unique = df[col].nunique()
        if n_unique > max_categories:
            logger.warning(f"Skipping '{col}': {n_unique} categories > max {max_categories}. "
                           f"Consider target encoding or label encoding instead.")
        else:
            to_encode.append(col)

    if to_encode:
        df = pd.get_dummies(df, columns=to_encode, drop_first=drop_first, dtype=float)
        logger.info(f"One-hot encoded: {to_encode} → {df.shape[1]} columns")
    return df


def ordinal_encode(
    df: pd.DataFrame,
    column: str,
    order: list,
) -> pd.DataFrame:
    """
    Encode an ordinal column according to a custom order.

    Use this for ordered categories where the numerical values carry meaning:
    "low" < "medium" < "high"
    "cold" < "warm" < "hot"

    Args:
        df:     DataFrame.
        column: Column to encode.
        order:  List of categories in ascending order.

    Returns:
        DataFrame with column replaced by integers.

    Example:
        df = ordinal_encode(df, "education",
                            order=["High School", "Bachelor", "Master", "PhD"])
        # Maps: HS→0, Bachelor→1, Master→2, PhD→3
    """
    df = df.copy()
    mapping = {val: i for i, val in enumerate(order)}
    unmapped = set(df[column].dropna().unique()) - set(order)
    if unmapped:
        logger.warning(f"'{column}': unseen categories will become NaN: {unmapped}")
    df[column] = df[column].map(mapping)
    logger.info(f"Ordinal encoded '{column}': {mapping}")
    return df


def target_encode(
    df: pd.DataFrame,
    column: str,
    target: str,
    smoothing: float = 1.0,
    train_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Target encoding: replace each category with the smoothed target mean.

    Best for high-cardinality categoricals (cities, zipcodes, user IDs)
    where one-hot encoding would create thousands of columns.

    Smoothing formula:
        encoded = (n × category_mean + k × global_mean) / (n + k)
    where n = category count and k = smoothing factor.

    WARNING: Must be computed only on training data and applied to test/val.
    Use train_mask to specify training rows.

    Args:
        df:         DataFrame.
        column:     Categorical column to encode.
        target:     Numeric target column.
        smoothing:  Regularization strength (higher = closer to global mean).
        train_mask: Boolean mask for training rows. If None, uses all rows.

    Returns:
        DataFrame with column replaced by target means.

    Example:
        df = target_encode(df, "city", "price", smoothing=5.0,
                           train_mask=df["split"] == "train")
    """
    df = df.copy()
    if train_mask is None:
        train_mask = pd.Series([True] * len(df), index=df.index)

    global_mean = df.loc[train_mask, target].mean()
    stats = df.loc[train_mask].groupby(column)[target].agg(["mean", "count"])
    stats["encoded"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )
    df[column] = df[column].map(stats["encoded"]).fillna(global_mean)
    logger.info(f"Target encoded '{column}' using '{target}' (smoothing={smoothing})")
    return df

#  INTERMEDIATE — Scaling
def scale_features(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "standard",
    clip_range: Optional[tuple[float, float]] = None,
) -> tuple[pd.DataFrame, Any]:
    """
    Scale numeric features.

    Scalers and when to use them:
        "standard":   Z-score: (x - mean) / std → centered at 0, std=1.
                      USE: Linear models, SVMs, neural networks, PCA.
                      NOT: Tree-based models (they're scale-invariant).

        "minmax":     Scale to [0, 1]: (x - min) / (max - min).
                      USE: Neural networks, image pixel values.
                      NOT: When you have outliers (they compress the range).

        "robust":     Scale using IQR instead of std.
                      USE: Data with outliers. More robust than standard.
                      NOT: When exact range matters.

        "maxabs":     Scale to [-1, 1] by max absolute value.
                      USE: Sparse data (preserves sparsity).

    Args:
        df:          DataFrame.
        columns:     Columns to scale. None = all numeric.
        method:      Scaler type.
        clip_range:  Optional (min, max) to clip after scaling.

    Returns:
        (scaled_df, fitted_scaler) — keep the scaler to transform test data!

    CRITICAL: Fit the scaler on TRAINING data only.
    Apply (transform) it to validation and test data. Never refit on test data.

    Example:
        # Training
        X_train_scaled, scaler = scale_features(X_train, method="standard")
        # Test — use the SAME scaler, just .transform()
        X_test[num_cols] = scaler.transform(X_test[num_cols])
    """
    from sklearn.preprocessing import MaxAbsScaler

    df = df.copy()
    num_cols = columns or df.select_dtypes(include=np.number).columns.tolist()

    scalers = {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
        "maxabs":   MaxAbsScaler(),
    }
    if method not in scalers:
        raise ValueError(f"Unknown scaler: {method}. Choose from {list(scalers.keys())}")

    scaler = scalers[method]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    if clip_range:
        lo, hi = clip_range
        df[num_cols] = df[num_cols].clip(lower=lo, upper=hi)

    logger.info(f"Scaled {len(num_cols)} columns with {method}Scaler")
    return df, scaler

#  ADVANCED — Feature engineering
def extract_datetime_features(
    df: pd.DataFrame,
    column: str,
    features: Optional[list[str]] = None,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Expand a datetime column into multiple numeric/categorical features.

    Datetime columns are rich with information that models can't use raw.
    This extracts that temporal signal into learnable features.

    Available features:
        "year", "month", "day", "hour", "minute", "second",
        "day_of_week",    0=Monday … 6=Sunday
        "day_of_year",    1–366
        "week_of_year",   1–53
        "quarter",        1–4
        "is_weekend",     0 or 1
        "is_month_start", 0 or 1
        "is_month_end",   0 or 1
        "sin_hour",       cyclic encoding (hour as sine)
        "cos_hour",       cyclic encoding (hour as cosine)
        "sin_month",      cyclic encoding (month as sine)
        "cos_month",      cyclic encoding (month as cosine)

    WHY CYCLIC ENCODING?
        Hour 23 and hour 0 are close in time, but far apart numerically.
        Encoding hour as sin/cos places them close together on a circle.

    Args:
        df:           DataFrame.
        column:       Datetime column name.
        features:     List of features to extract. None = a good default set.
        drop_original: Remove the original datetime column.

    Returns:
        DataFrame with new time features.

    Example:
        df = cast_dtypes(df, {"created_at": "datetime64[ns]"})
        df = extract_datetime_features(df, "created_at")
    """
    df = df.copy()
    dt = pd.to_datetime(df[column])

    if features is None:
        features = ["year", "month", "day", "hour", "day_of_week",
                    "is_weekend", "sin_hour", "cos_hour", "sin_month", "cos_month"]

    feature_map = {
        "year":          lambda: dt.dt.year,
        "month":         lambda: dt.dt.month,
        "day":           lambda: dt.dt.day,
        "hour":          lambda: dt.dt.hour,
        "minute":        lambda: dt.dt.minute,
        "second":        lambda: dt.dt.second,
        "day_of_week":   lambda: dt.dt.dayofweek,
        "day_of_year":   lambda: dt.dt.dayofyear,
        "week_of_year":  lambda: dt.dt.isocalendar().week.astype(int),
        "quarter":       lambda: dt.dt.quarter,
        "is_weekend":    lambda: (dt.dt.dayofweek >= 5).astype(int),
        "is_month_start":lambda: dt.dt.is_month_start.astype(int),
        "is_month_end":  lambda: dt.dt.is_month_end.astype(int),
        # Cyclic: sin and cos so that 23:00 and 00:00 are close
        "sin_hour":      lambda: np.sin(2 * np.pi * dt.dt.hour / 24),
        "cos_hour":      lambda: np.cos(2 * np.pi * dt.dt.hour / 24),
        "sin_month":     lambda: np.sin(2 * np.pi * dt.dt.month / 12),
        "cos_month":     lambda: np.cos(2 * np.pi * dt.dt.month / 12),
    }

    for feat in features:
        if feat in feature_map:
            df[f"{column}_{feat}"] = feature_map[feat]()
        else:
            logger.warning(f"Unknown datetime feature: {feat}")

    if drop_original:
        df = df.drop(columns=[column])

    logger.info(f"Extracted {len(features)} datetime features from '{column}'")
    return df


def create_interaction_features(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    operations: list[str] = ("multiply",),
) -> pd.DataFrame:
    """
    Create interaction features between pairs of columns.

    Interaction features capture non-linear relationships that models
    (especially linear ones) might otherwise miss.

    Operations:
        "multiply": col_a × col_b
        "divide":   col_a / col_b
        "add":      col_a + col_b
        "subtract": col_a - col_b

    Args:
        df:         DataFrame.
        pairs:      List of (col_a, col_b) tuples.
        operations: Which operations to apply.

    Returns:
        DataFrame with new interaction columns appended.

    Example:
        df = create_interaction_features(df,
            pairs=[("price", "quantity"), ("age", "salary")],
            operations=["multiply", "divide"],
        )
        # Creates: price_x_quantity, price_div_quantity, age_x_salary, ...
    """
    df = df.copy()
    op_map = {
        "multiply": lambda a, b: a * b,
        "divide":   lambda a, b: a / (b + 1e-8),
        "add":      lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
    }
    created = []
    for col_a, col_b in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            logger.warning(f"Skipping ({col_a}, {col_b}) — column not found")
            continue
        for op in operations:
            suffix = {"multiply": "x", "divide": "div", "add": "plus", "subtract": "minus"}[op]
            name = f"{col_a}_{suffix}_{col_b}"
            df[name] = op_map[op](df[col_a], df[col_b])
            created.append(name)

    logger.info(f"Created {len(created)} interaction features")
    return df


def drop_low_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Remove columns with variance below threshold.

    Low-variance features carry little information — they're almost constant.
    Such features waste model capacity and may harm generalization.

    Note: Scale your data first — variance is scale-dependent!

    Args:
        df:        DataFrame.
        threshold: Minimum variance. Columns below this are dropped.
        columns:   Subset to check. None = all numeric.

    Returns:
        DataFrame with low-variance columns removed.

    Example:
        df = scale_features(df, method="minmax")[0]  # scale first
        df = drop_low_variance(df, threshold=0.01)
    """
    from sklearn.feature_selection import VarianceThreshold
    df = df.copy()
    num_cols = columns or df.select_dtypes(include=np.number).columns.tolist()

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[num_cols])
    to_keep = [col for col, keep in zip(num_cols, selector.get_support()) if keep]
    to_drop = [col for col in num_cols if col not in to_keep]

    if to_drop:
        logger.info(f"Dropping {len(to_drop)} low-variance columns: {to_drop}")
    return df.drop(columns=to_drop)


def drop_correlated(
    df: pd.DataFrame,
    threshold: float = 0.95,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Remove highly correlated numeric columns.

    When two features are almost perfectly correlated, one is redundant.
    Keeping both can cause multicollinearity in linear models.

    Strategy: Compute pairwise correlation matrix, keep one from each pair
    where |correlation| > threshold.

    Args:
        df:        DataFrame.
        threshold: |Pearson correlation| threshold. 0.95 = very conservative.
        columns:   Subset of columns. None = all numeric.

    Returns:
        DataFrame without redundant correlated columns.

    Example:
        df = drop_correlated(df, threshold=0.90)
    """
    df = df.copy()
    num_cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr().abs()

    # Upper triangle mask
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        logger.info(f"Dropping {len(to_drop)} correlated columns (threshold={threshold}): {to_drop}")
    return df.drop(columns=to_drop)

#  ADVANCED — Train/Val/Test Split
def split_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    WHY THREE SPLITS?
        - Train: Model learns from this.
        - Validation: Tune hyperparameters (without touching test).
        - Test: Final honest evaluation — NEVER touch during development.

    WHY STRATIFY?
        Ensures class proportions are preserved in each split.
        Critical for imbalanced datasets. (Only for classification.)

    Args:
        df:           Full DataFrame.
        target:       Name of the target column.
        test_size:    Fraction for test set (default 0.2 = 20%).
        val_size:     Fraction for validation set (default 0.1 = 10%).
        stratify:     Stratify by target. Set False for regression.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test

    Example:
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_dataset(df, target="label")
        print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")
    """
    X = df.drop(columns=[target])
    y = df[target]

    strat = y if stratify and y.nunique() < 50 else None

    # First: carve out test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # Then: carve val from remaining
    val_frac = val_size / (1 - test_size)
    strat_temp = y_temp if strat is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_frac,
        random_state=random_state,
        stratify=strat_temp,
    )

    total = len(df)
    logger.info(f"Split: train={len(X_train)} ({len(X_train)/total:.0%}), "
                f"val={len(X_val)} ({len(X_val)/total:.0%}), "
                f"test={len(X_test)} ({len(X_test)/total:.0%})")
    return X_train, X_val, X_test, y_train, y_val, y_test

#  PIPELINE — Compose everything
class StructuredPreprocessingPipeline:
    """
    Full configurable preprocessing pipeline for tabular data.

    Steps (in order, each optional):
      1. Remove duplicates
      2. Drop high-missing columns
      3. Clip outliers
      4. Impute missing values
      5. Encode categoricals (one-hot or ordinal)
      6. Scale numerics
      7. Drop low-variance features
      8. Drop correlated features

    Example (beginner — minimal cleaning):
        pipeline = StructuredPreprocessingPipeline(
            target="price",
            impute_strategy="median",
            scale_method="standard",
        )
        X_train_proc, y_train = pipeline.fit_transform(df_train)
        X_test_proc = pipeline.transform(df_test)

    Example (full config):
        pipeline = StructuredPreprocessingPipeline(
            target="label",
            remove_dups=True,
            drop_missing_threshold=0.4,
            clip_outliers_method="iqr",
            impute_strategy="median",
            cat_columns=["color", "size"],
            encode_method="onehot",
            scale_method="robust",
            drop_variance_threshold=0.01,
            drop_correlation_threshold=0.95,
        )
    """

    def __init__(
        self,
        target: str,
        # Cleaning
        remove_dups: bool = True,
        drop_missing_threshold: float = 0.5,
        clip_outliers_method: Optional[str] = "iqr",
        clip_factor: float = 1.5,
        # Imputation
        impute_strategy: str = "median",
        # Encoding
        cat_columns: Optional[list[str]] = None,
        encode_method: str = "onehot",      # "onehot" | "label" | "none"
        ordinal_orders: Optional[dict[str, list]] = None,
        # Scaling
        scale_method: Optional[str] = "standard",
        # Feature selection
        drop_variance_threshold: Optional[float] = None,
        drop_correlation_threshold: Optional[float] = None,
    ):
        self.target                    = target
        self.remove_dups               = remove_dups
        self.drop_missing_threshold    = drop_missing_threshold
        self.clip_outliers_method      = clip_outliers_method
        self.clip_factor               = clip_factor
        self.impute_strategy           = impute_strategy
        self.cat_columns               = cat_columns or []
        self.encode_method             = encode_method
        self.ordinal_orders            = ordinal_orders or {}
        self.scale_method              = scale_method
        self.drop_variance_threshold   = drop_variance_threshold
        self.drop_correlation_threshold = drop_correlation_threshold
        self._scaler                   = None
        self._fitted                   = False

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on training data and return transformed X, y."""
        df = df.copy()
        y = df.pop(self.target)

        if self.remove_dups:
            df = remove_duplicates(df)
            y = y.loc[df.index]

        df = drop_high_missing(df, self.drop_missing_threshold)

        if self.clip_outliers_method:
            df = clip_outliers(df, method=self.clip_outliers_method, factor=self.clip_factor)

        df = impute(df, strategy=self.impute_strategy)

        for col, order in self.ordinal_orders.items():
            if col in df.columns:
                df = ordinal_encode(df, col, order)

        cat = [c for c in self.cat_columns if c in df.columns and c not in self.ordinal_orders]
        if cat and self.encode_method == "onehot":
            df = one_hot_encode(df, cat)
        elif cat and self.encode_method == "label":
            for col in cat:
                df, _ = encode_labels(df, col)

        if self.scale_method:
            df, self._scaler = scale_features(df, method=self.scale_method)

        if self.drop_variance_threshold is not None:
            df = drop_low_variance(df, self.drop_variance_threshold)

        if self.drop_correlation_threshold is not None:
            df = drop_correlated(df, self.drop_correlation_threshold)

        self._fitted_columns = df.columns.tolist()
        self._fitted = True
        logger.info(f"Pipeline fit complete. Output: {df.shape}")
        return df, y.reset_index(drop=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted pipeline to new data (val/test)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        df = df.copy()
        if self.target in df.columns:
            df = df.drop(columns=[self.target])

        df = impute(df, strategy=self.impute_strategy)

        for col, order in self.ordinal_orders.items():
            if col in df.columns:
                df = ordinal_encode(df, col, order)

        cat = [c for c in self.cat_columns if c in df.columns and c not in self.ordinal_orders]
        if cat and self.encode_method == "onehot":
            df = one_hot_encode(df, cat)

        if self._scaler is not None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            df[num_cols] = self._scaler.transform(df[num_cols])

        # Align columns with training set
        for col in self._fitted_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in self._fitted_columns if c in df.columns]]

        return df

#  QUICK-START DEMO
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("  structured_preprocessing.py — Demo")
    print("=" * 60)

    # Create messy synthetic data
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age":        np.random.randint(18, 70, n).astype(float),
        "salary":     np.random.lognormal(10.5, 0.5, n),
        "department": np.random.choice(["Eng", "HR", "Sales", None], n),
        "education":  np.random.choice(["High School", "Bachelor", "Master", "PhD"], n),
        "score":      np.random.normal(0, 1, n),
        "label":      np.random.choice([0, 1], n),
    })
    # Inject missing values
    df.loc[np.random.choice(n, 20), "age"] = np.nan
    df.loc[np.random.choice(n, 15), "salary"] = np.nan
    # Inject outliers
    df.loc[0, "salary"] = 10_000_000
    # Add duplicate
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    print(f"\n Original: {df.shape}")
    print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Profile
    print("\n📊 Data profile (first 4 columns):")
    prof = profile(df, target="label")
    print(prof[["dtype", "n_missing", "missing_pct", "n_unique"]].head(4))

    # Step-by-step (pedagogic)
    print("\n🟡 Step-by-step:")
    df2 = remove_duplicates(df)
    df2 = clip_outliers(df2, columns=["salary"], method="iqr")
    df2 = impute(df2, strategy={"age": "median", "salary": "median", "department": "mode"})
    df2 = ordinal_encode(df2, "education", ["High School", "Bachelor", "Master", "PhD"])
    df2 = one_hot_encode(df2, ["department"])
    df2, scaler = scale_features(df2.drop(columns=["label"]), method="standard")
    print(f"   After processing: {df2.shape}")

    # Pipeline
    print("\n Full pipeline:")
    pipeline = StructuredPreprocessingPipeline(
        target="label",
        cat_columns=["department"],
        ordinal_orders={"education": ["High School", "Bachelor", "Master", "PhD"]},
        scale_method="standard",
        clip_outliers_method="iqr",
    )
    X_train_proc, y_train = pipeline.fit_transform(df)
    print(f"   Pipeline output: {X_train_proc.shape}")
    print(f"   Features: {X_train_proc.columns.tolist()}")

    # Datetime features demo
    print("\n⏱  Datetime feature extraction:")
    df_time = pd.DataFrame({
        "event_time": pd.date_range("2024-01-01", periods=5, freq="6H"),
        "value": [10, 20, 15, 25, 30],
    })
    df_time = extract_datetime_features(df_time, "event_time")
    print(f"   Extracted columns: {df_time.columns.tolist()}")

    print("\n✅ Structured preprocessing demo complete.")