"""
╔══════════════════════════════════════════════════════════════════════╗
║               synthetic_data.py — Synthetic Data Generation          ║
║                                                                      ║
║  Generate realistic fake data for testing, prototyping, privacy-     ║
║  safe development, and ML dataset bootstrapping.                     ║
║                                                                      ║
║  Levels covered:                                                     ║
║     Beginner  — Faker-based profiles, names, addresses               ║
║      Tabular datasets, time series, distributions      ║
║     — Correlated features, constraint-aware generation     ║
║                                                                      ║
║  Dependencies:                                                       ║
║    pip install faker numpy pandas scipy                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import random
import logging
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from faker import Faker

logger = logging.getLogger("synthetic_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# One global Faker instance. Seed it for reproducibility.
_faker = Faker()

#  SEED CONTROL — Reproducibility is essential in data work
def set_seed(seed: int) -> None:
    """
    Fix all random seeds so generated data is reproducible.

    ALWAYS do this if you need the same dataset each run,
    e.g., for unit tests or published experiments.

    Args:
        seed: Any integer. Common choices: 42, 0, 2024.

    Example:
        set_seed(42)
        df = generate_user_profiles(100)  # same 100 rows every time
    """
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)
    logger.info(f"Random seed set to {seed}")

#  BEGINNER — Pre-built realistic entity generators
def generate_user_profiles(
    n: int = 100,
    locale: Union[str, list[str]] = "en_US",
) -> list[dict]:
    """
    Generate n realistic user profile records.

    Each profile includes: name, email, phone, address, DOB, job, company.

    Args:
        n:      Number of profiles to generate.
        locale: Faker locale for localized data. Examples:
                "fr_FR" (French), "de_DE" (German), "ja_JP" (Japanese),
                or ["en_US", "fr_FR"] for mixed locales.

    Returns:
        List of dicts.

    Example:
        profiles = generate_user_profiles(50, locale="fr_FR")
        for p in profiles[:3]:
            print(p["name"], p["email"])
    """
    fake = Faker(locale)
    profiles = []
    for _ in range(n):
        dob = fake.date_of_birth(minimum_age=18, maximum_age=75)
        profiles.append({
            "id":         fake.uuid4(),
            "name":       fake.name(),
            "email":      fake.email(),
            "phone":      fake.phone_number(),
            "address":    fake.address().replace("\n", ", "),
            "city":       fake.city(),
            "country":    fake.country(),
            "dob":        dob.isoformat(),
            "age":        (datetime.today().date() - dob).days // 365,
            "job":        fake.job(),
            "company":    fake.company(),
            "username":   fake.user_name(),
            "created_at": fake.date_time_this_decade().isoformat(),
        })
    logger.info(f"Generated {n} user profiles (locale={locale})")
    return profiles


def generate_product_catalog(n: int = 50) -> list[dict]:
    """
    Generate a fake e-commerce product catalog.

    Returns:
        List of product dicts with: id, name, category, price, stock, rating.

    Example:
        products = generate_product_catalog(100)
        df = pd.DataFrame(products)
        print(df.describe())
    """
    fake = Faker()
    categories = ["Electronics", "Books", "Clothing", "Home & Garden",
                  "Sports", "Toys", "Beauty", "Food & Drink", "Tools"]
    products = []
    for i in range(n):
        category = random.choice(categories)
        products.append({
            "product_id":  f"SKU-{i:05d}",
            "name":        f"{fake.word().title()} {fake.word().title()}",
            "category":    category,
            "price":       round(random.uniform(1.99, 999.99), 2),
            "cost":        round(random.uniform(0.50, 400.00), 2),
            "stock":       random.randint(0, 500),
            "rating":      round(random.uniform(1.0, 5.0), 1),
            "num_reviews": random.randint(0, 5000),
            "in_stock":    random.random() > 0.15,
            "created_at":  fake.date_this_decade().isoformat(),
            "description": fake.sentence(nb_words=15),
        })
    logger.info(f"Generated {n} product records")
    return products


def generate_transactions(
    n: int = 500,
    user_ids: Optional[list[str]] = None,
    product_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Generate fake purchase transactions.

    Can be linked to real user/product IDs for relational coherence.

    Args:
        n:           Number of transactions.
        user_ids:    Pool of user IDs to sample from.
        product_ids: Pool of product IDs to sample from.

    Example:
        users    = generate_user_profiles(100)
        products = generate_product_catalog(50)
        txns     = generate_transactions(1000,
                       user_ids=[u["id"] for u in users],
                       product_ids=[p["product_id"] for p in products])
    """
    fake = Faker()
    uid_pool = user_ids or [fake.uuid4() for _ in range(50)]
    pid_pool = product_ids or [f"SKU-{i:05d}" for i in range(30)]
    payment_methods = ["credit_card", "debit_card", "paypal", "crypto", "bank_transfer"]
    statuses = ["completed", "pending", "refunded", "failed"]
    status_weights = [0.80, 0.10, 0.07, 0.03]

    txns = []
    for i in range(n):
        #Discrete uniform distribution
        qty = random.randint(1, 5)
        unit_price = round(random.uniform(1.99, 299.99), 2)
        txns.append({
            "transaction_id": fake.uuid4(),
            "user_id":        random.choice(uid_pool),
            "product_id":     random.choice(pid_pool),
            "quantity":       qty,
            "unit_price":     unit_price,
            "total":          round(qty * unit_price, 2),
            "status":         random.choices(statuses, weights=status_weights)[0],
            "payment_method": random.choice(payment_methods),
            "timestamp":      fake.date_time_this_year().isoformat(),
            "ip_address":     fake.ipv4(),
        })
    logger.info(f"Generated {n} transactions")
    return txns

#   Numerical and statistical data
def generate_time_series(
    n_points: int = 365,
    start_date: str = "2023-01-01",
    freq: str = "D",
    trend: float = 0.05,
    seasonality_period: int = 7,
    seasonality_amplitude: float = 10.0,
    noise_std: float = 5.0,
    base_value: float = 100.0,
) -> pd.DataFrame:
    """
    Generate a realistic time series with trend + seasonality + noise.

    This decomposition (Trend + Seasonality + Residual) is the foundation
    of classical time series analysis (see: STL decomposition, SARIMA).

    Args:
        n_points:             Number of time steps.
        start_date:           First date in the series.
        freq:                 Pandas frequency string. "D"=daily, "H"=hourly, "W"=weekly.
        trend:                Linear slope per time step.
        seasonality_period:   Repeating cycle length (7=weekly, 12=monthly).
        seasonality_amplitude: Peak-to-peak variation of the seasonal component.
        noise_std:            Standard deviation of random noise.
        base_value:           Starting value.

    Returns:
        DataFrame with columns: date, value, trend_component,
        seasonal_component, noise_component.

    Example:
        ts = generate_time_series(365, seasonality_period=7, trend=0.1)
        ts.plot(x="date", y="value")
    """
    dates = pd.date_range(start=start_date, periods=n_points, freq=freq)
    t = np.arange(n_points)

    trend_component     = base_value + trend * t
    seasonal_component  = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
    noise_component     = np.random.normal(0, noise_std, n_points)

    value = trend_component + seasonal_component + noise_component

    df = pd.DataFrame({
        "date":               dates,
        "value":              value,
        "trend_component":    trend_component,
        "seasonal_component": seasonal_component,
        "noise_component":    noise_component,
    })
    logger.info(f"Generated time series with {n_points} points")
    return df


def generate_tabular_dataset(
    n_rows: int = 1000,
    schema: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    Generate a tabular dataset from a declarative column schema.

    The schema lets you define each column's type, distribution, and
    any specific parameters. This is the most flexible generator here.

    Supported column types:
        - "normal":      Gaussian distribution. Params: mean, std.
        - "uniform":     Uniform distribution. Params: low, high.
        - "lognormal":   Log-normal (always positive). Params: mean, sigma.
        - "exponential": Exponential decay. Params: scale.
        - "categorical": Random choice from values list. Params: values, weights.
        - "boolean":     True/False. Params: p_true (probability of True).
        - "integer":     Random integers. Params: low, high.
        - "datetime":    Random dates. Params: start, end (YYYY-MM-DD strings).
        - "id":          Auto-incrementing integer ID (no extra params needed).

    Args:
        n_rows:  Number of rows.
        schema:  Dict of column_name → {type, …params}. If None, a
                 default HR-style dataset is generated as an example.

    Returns:
        DataFrame.

    Example:
        schema = {
            "age":         {"type": "integer", "low": 18, "high": 65},
            "salary":      {"type": "lognormal", "mean": 10.5, "sigma": 0.5},
            "department":  {"type": "categorical",
                            "values": ["Eng", "HR", "Sales", "Ops"],
                            "weights": [0.40, 0.15, 0.30, 0.15]},
            "is_remote":   {"type": "boolean", "p_true": 0.6},
        }
        df = generate_tabular_dataset(500, schema)
    """
    if schema is None:
        # Default example: employee dataset
        schema = {
            "employee_id":  {"type": "id"},
            "age":          {"type": "integer", "low": 22, "high": 65},
            "salary":       {"type": "lognormal", "mean": 10.8, "sigma": 0.4},
            "years_exp":    {"type": "integer", "low": 0, "high": 40},
            "department":   {"type": "categorical",
                             "values": ["Engineering", "Marketing", "Sales",
                                        "HR", "Finance", "Operations"],
                             "weights": [0.35, 0.15, 0.20, 0.10, 0.10, 0.10]},
            "is_remote":    {"type": "boolean", "p_true": 0.55},
            "perf_score":   {"type": "normal", "mean": 3.5, "std": 0.8},
            "hire_date":    {"type": "datetime", "start": "2010-01-01", "end": "2024-01-01"},
        }

    data = {}
    for col, cfg in schema.items():
        col_type = cfg["type"]
        if col_type == "normal":
            data[col] = np.random.normal(cfg.get("mean", 0), cfg.get("std", 1), n_rows)
        elif col_type == "uniform":
            data[col] = np.random.uniform(cfg.get("low", 0), cfg.get("high", 1), n_rows)
        elif col_type == "lognormal":
            data[col] = np.random.lognormal(cfg.get("mean", 0), cfg.get("sigma", 1), n_rows)
        elif col_type == "exponential":
            data[col] = np.random.exponential(cfg.get("scale", 1), n_rows)
        elif col_type == "categorical":
            values  = cfg["values"]
            weights = cfg.get("weights", None)
            data[col] = random.choices(values, weights=weights, k=n_rows)
        elif col_type == "boolean":
            data[col] = np.random.random(n_rows) < cfg.get("p_true", 0.5)
        elif col_type == "integer":
            data[col] = np.random.randint(cfg.get("low", 0), cfg.get("high", 100) + 1, n_rows)
        elif col_type == "datetime":
            start = pd.Timestamp(cfg.get("start", "2020-01-01"))
            end   = pd.Timestamp(cfg.get("end",   "2024-01-01"))
            delta = (end - start).days
            data[col] = [start + timedelta(days=random.randint(0, delta)) for _ in range(n_rows)]
        elif col_type == "id":
            data[col] = list(range(1, n_rows + 1))
        else:
            raise ValueError(f"Unknown column type: '{col_type}'")

    df = pd.DataFrame(data)
    logger.info(f"Generated tabular dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df

#   Correlated & constrained data generation
def generate_correlated_features(
    n: int = 1000,
    correlation_matrix: Optional[np.ndarray] = None,
    means: Optional[list[float]] = None,
    stds: Optional[list[float]] = None,
    column_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate multiple numeric features with specified correlations.

    Real-world data has correlated features (e.g., height & weight).
    This function uses Cholesky decomposition to enforce correlations.

    HOW IT WORKS (math explained simply):
      1. Generate independent standard normal vectors.
      2. Apply a Cholesky factor L (from the correlation matrix R = L·Lᵀ)
         to mix the independent vectors → correlated vectors.
      3. Scale by desired means and stds.

    Args:
        n:                  Number of samples.
        correlation_matrix: Square symmetric matrix where entry (i,j) is the
                           desired Pearson correlation between features i and j.
                           Diagonal must be 1. If None, uses a demo 3-feature matrix.
        means:              Mean for each feature. Defaults to 0.
        stds:               Std dev for each feature. Defaults to 1.
        column_names:       Column labels. Auto-named if None.

    Returns:
        DataFrame with correlated columns.

    Example:
        # height, weight, shoe_size should be positively correlated
        R = np.array([
            [1.0,  0.8,  0.7],
            [0.8,  1.0,  0.75],
            [0.7,  0.75, 1.0],
        ])
        df = generate_correlated_features(500, R,
                means=[170, 70, 42], stds=[10, 15, 3],
                column_names=["height_cm", "weight_kg", "shoe_size"])
        print(df.corr())   # should be close to R
    """
    if correlation_matrix is None:
        correlation_matrix = np.array([
            [1.0,  0.7, -0.3],
            [0.7,  1.0, -0.2],
            [-0.3, -0.2,  1.0],
        ])

    k = correlation_matrix.shape[0]
    means = means or [0.0] * k
    stds  = stds  or [1.0] * k
    names = column_names or [f"feature_{i}" for i in range(k)]

    # Cholesky decomposition — the key math step
    L = np.linalg.cholesky(correlation_matrix)
    # Independent standard normals
    Z = np.random.randn(n, k)
    # Introduce correlations: each row of Z·Lᵀ is now correlated
    correlated = Z @ L.T
    # Scale to desired means and stds
    for i in range(k):
        correlated[:, i] = correlated[:, i] * stds[i] + means[i]

    df = pd.DataFrame(correlated, columns=names)
    logger.info(f"Generated {k}-feature correlated dataset (n={n})")
    logger.info(f"Realized correlations:\n{df.corr().round(2)}")
    return df


def generate_classification_dataset(
    n: int = 1000,
    n_classes: int = 2,
    n_features: int = 10,
    n_informative: int = 5,
    class_weights: Optional[list[float]] = None,
    noise: float = 0.1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a labeled classification dataset.

    Informative features truly differ between classes.
    Non-informative features are pure noise — this tests whether
    your ML model can correctly ignore them.

    Args:
        n:             Total samples.
        n_classes:     Number of target classes (2 = binary, 3+ = multiclass).
        n_features:    Total number of features (including uninformative ones).
        n_informative: How many features actually carry class signal.
        class_weights: Relative frequency of each class. None = balanced.
                       e.g., [0.9, 0.1] for an imbalanced binary problem.
        noise:         Overlap / noise level between classes (0=clean, 1=random).

    Returns:
        (X DataFrame, y Series)

    Example:
        X, y = generate_classification_dataset(2000, n_classes=3, n_informative=6)
        # Plug directly into sklearn:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier().fit(X, y)
    """
    from scipy.stats import multivariate_normal  # lazy import

    if class_weights is None:
        class_weights = [1 / n_classes] * n_classes
    # Normalize
    total = sum(class_weights)
    class_weights = [w / total for w in class_weights]

    class_sizes = [int(n * w) for w in class_weights]
    class_sizes[-1] = n - sum(class_sizes[:-1])   # fix rounding

    X_parts, y_parts = [], []
    for cls, size in enumerate(class_sizes):
        # Informative features: class-specific means, spread by noise
        means = np.random.randn(n_informative) * 2
        cov   = np.eye(n_informative) * (1 + noise * 3)
        info  = multivariate_normal.rvs(mean=means, cov=cov, size=size)
        # Uninformative features: pure noise, same for all classes
        noisy = np.random.randn(size, n_features - n_informative)
        X_cls = np.hstack([info.reshape(size, -1), noisy])
        X_parts.append(X_cls)
        y_parts.extend([cls] * size)

    X = np.vstack(X_parts)
    y = np.array(y_parts)
    # Shuffle
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]

    col_names = [f"feat_{i:02d}" for i in range(n_features)]
    df_X = pd.DataFrame(X, columns=col_names)
    s_y  = pd.Series(y, name="label")
    logger.info(f"Classification dataset: {n} samples, {n_classes} classes, "
                f"{n_informative}/{n_features} informative features")
    return df_X, s_y


def inject_missing_values(
    df: pd.DataFrame,
    missing_rate: float = 0.05,
    strategy: str = "mcar",
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Deliberately introduce missing values into a DataFrame.

    Why? To test your imputation pipelines, evaluate robustness,
    and simulate real-world messy data.

    Strategies (from statistics on missing data):
        - "mcar": Missing Completely At Random — simplest, no pattern.
        - "mar":  Missing At Random — missingness depends on other cols.
        - "mnar": Missing Not At Random — values missing based on own value.

    Args:
        df:           Input DataFrame.
        missing_rate: Fraction of values to set to NaN (0–1).
        strategy:     "mcar" | "mar" | "mnar"
        columns:      Columns to apply to. None = all numeric columns.

    Returns:
        DataFrame copy with injected NaN values.

    Example:
        dirty_df = inject_missing_values(df, missing_rate=0.1, strategy="mcar")
        print(dirty_df.isnull().mean())
    """
    df = df.copy()
    target_cols = columns or df.select_dtypes(include=np.number).columns.tolist()

    for col in target_cols:
        n = len(df)
        if strategy == "mcar":
            mask = np.random.random(n) < missing_rate
        elif strategy == "mar":
            # Missingness depends on another random column's value
            ref_col = random.choice([c for c in target_cols if c != col] or [col])
            threshold = df[ref_col].quantile(missing_rate)
            mask = df[ref_col] < threshold
        elif strategy == "mnar":
            # Values more likely missing if they're extreme (top or bottom)
            threshold_hi = df[col].quantile(1 - missing_rate / 2)
            threshold_lo = df[col].quantile(missing_rate / 2)
            mask = (df[col] > threshold_hi) | (df[col] < threshold_lo)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        df.loc[mask, col] = np.nan

    pct = df.isnull().mean().mean()
    logger.info(f"Injected missing values (strategy={strategy}, actual rate={pct:.2%})")
    return df

#  EXPORT — Save to multiple formats
def save_dataset(
    data: Union[pd.DataFrame, list[dict]],
    filepath: str,
    fmt: str = "csv",
) -> None:
    """
    Save synthetic data to disk in csv, json, or parquet format.

    Args:
        data:     DataFrame or list of dicts.
        filepath: Output path (extension auto-added if missing).
        fmt:      "csv" | "json" | "parquet" | "excel"

    Example:
        df = generate_tabular_dataset(1000)
        save_dataset(df, "output/employees", fmt="csv")
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, list):
        data = pd.DataFrame(data)

    if fmt == "csv":
        out = path.with_suffix(".csv")
        data.to_csv(out, index=False)
    elif fmt == "json":
        out = path.with_suffix(".json")
        data.to_json(out, orient="records", indent=2, default_handler=str, date_format="iso", date_unit="s")
    elif fmt == "parquet":
        out = path.with_suffix(".parquet")
        data.to_parquet(out, index=False)
    elif fmt == "excel":
        out = path.with_suffix(".xlsx")
        data.to_excel(out, index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}. Use csv, json, parquet, or excel.")

    logger.info(f"Saved {len(data)} rows to {out}")

# QUICK-START DEMO
if __name__ == "__main__":
    set_seed(42)
    Path("output").mkdir(exist_ok=True)

    print("\n Beginner: Generating user profiles…")
    users = generate_user_profiles(20)
    print(f"   Sample: {users[0]['name']} | {users[0]['email']}")
    save_dataset(users, "output/users", fmt="csv")

    print("\n Generating time series…")
    ts = generate_time_series(180, seasonality_period=7, trend=0.2, noise_std=8)
    print(ts.head(3))
    save_dataset(ts, "output/time_series", fmt="csv")

    print("\n Generating tabular dataset…")
    df = generate_tabular_dataset(500)
    print(df.head(3))
    save_dataset(df, "output/employees", fmt="csv")

    print("\n Generating correlated features…")
    R = np.array([[1.0, 0.8, -0.4],
                  [0.8, 1.0, -0.3],
                  [-0.4, -0.3, 1.0]])
    corr_df = generate_correlated_features(300, R, means=[170, 70, 42],
                                           stds=[10, 15, 3],
                                           column_names=["height", "weight", "shoe"])
    print("Correlation check:")
    print(corr_df.corr().round(2))

    print("\n Classification dataset + missing values…")
    X, y = generate_classification_dataset(500, n_classes=3, n_informative=4)
    X_dirty = inject_missing_values(X, missing_rate=0.08)
    print(f"   Class distribution:\n{y.value_counts()}")
    print(f"   Missing rate per column:\n{X_dirty.isnull().mean().round(3)}")

    print("\n✅ All synthetic data generated. See output/ directory.")