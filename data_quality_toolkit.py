"""
Unified Data Quality, PII Detection, K-Anonymity, AI Kosh Scoring &
Dataset Fitness Toolkit
===========================================================================

Merges three previously separate notebooks into one end-to-end pipeline:

  1. Data Quality Report Generator
     - Row/column counts, null counts & percentages, duplicate rows,
       per-column profiling (dtype, unique count, sample values).

  2. PII Detection Engine + K-Anonymity Engine
     - PII detection by column-name keywords AND by regex pattern
       matching on values (email, phone, Aadhaar, PAN, passport).
     - K-Anonymity: auto-detects quasi-identifier columns, computes
       group sizes, minimum k, and flags groups that violate a
       configurable k-threshold.

  3. AI Kosh Data Quality Scoring Framework (0-60 scale)
     - Data Volume / Missing Values / Schema Consistency / Uniqueness /
       Outliers / Entropy (Freedman-Diaconis adaptive binning)
     - Quality decision: GO (>=42) / CONDITIONAL GO (>=36) / NO GO (<36)

  4. Combined Dataset Fitness (new - ties 2 and 3 together)
     - Quality decision + k-anonymity privacy status are combined into
       one manager-facing verdict: FIT / CONDITIONAL / NOT FIT, with a
       plain-English reason. This was never actually computed in the
       original notebooks - each piece existed separately.

  5. Cross-Dataset Similarity & Duplication (schema overlap, business-key
     overlap, exact row-hash matches) across all input tables.

  6. Visualization (new)
     - Score-breakdown bar chart and top-missing-value-columns chart per
       table, embedded directly into each Excel report.
     - Batch-level fitness distribution chart in the master report.

Usage
-----
    python data_quality_toolkit.py --input <file_or_folder> --output <folder>

Run `python data_quality_toolkit.py --help` for all options.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless - no display needed
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image as XLImage


# =====================================================================
# 0. LOGGING
# =====================================================================
def setup_logger(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("dq_toolkit")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


# =====================================================================
# 1. FILE LOADING  (csv / xls / xlsx / parquet, single or multi-sheet)
# =====================================================================
SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx", ".parquet"}


def load_file(file_path: Path, sample_rows: int | None = None) -> dict:
    """Load a single data file. Returns {sheet_name: DataFrame}."""
    ext = file_path.suffix.lower()
    sheets: dict = {}

    if ext == ".csv":
        df, last_exc = None, None
        # utf-8-sig also reads plain UTF-8; cp1252/latin-1 rescue Windows exports.
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(file_path, low_memory=False, nrows=sample_rows, encoding=enc)
                break
            except UnicodeDecodeError as exc:
                last_exc = exc
        if df is None:
            raise last_exc
        sheets["NA"] = df

    elif ext in (".xls", ".xlsx"):
        sheets = pd.read_excel(file_path, sheet_name=None, nrows=sample_rows)

    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
        if sample_rows:
            df = df.head(sample_rows)
        sheets["NA"] = df

    else:
        raise ValueError(f"Unsupported extension: {ext}")

    return sheets


# =====================================================================
# 1b. EXCEL SAFETY HELPERS
# =====================================================================
# openpyxl raises IllegalCharacterError on control characters, and Excel
# cells cannot exceed 32,767 characters / 1,048,576 rows. One bad cell used
# to abort the whole batch at report-writing time.
_ILLEGAL_XL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EXCEL_MAX_CELL_CHARS = 32_000
EXCEL_SHEET_MAX_ROWS = 100_000   # cap for very large detail sheets (k-anonymity groups, cross keys)


def _clean_text(value):
    if not isinstance(value, str):
        return value
    return _ILLEGAL_XL_RE.sub("", value)[:EXCEL_MAX_CELL_CHARS]


def _xl_safe_df(df: pd.DataFrame, sort_col: str | None = None, label: str = "") -> pd.DataFrame:
    """Return a copy that is safe to write to Excel (clean text, capped rows)."""
    if df is None or df.empty:
        return df
    out = df
    if len(out) > EXCEL_SHEET_MAX_ROWS:
        LOGGER.warning("%s has %d rows; writing the first %d to Excel.",
                       label or "Detail sheet", len(out), EXCEL_SHEET_MAX_ROWS)
        if sort_col and sort_col in out.columns:
            out = out.sort_values(sort_col, kind="stable")
        out = out.head(EXCEL_SHEET_MAX_ROWS)
    out = out.copy()
    out.columns = [_clean_text(c) for c in out.columns]
    for i in range(out.shape[1]):
        col = out.iloc[:, i]
        if col.dtype == object or str(col.dtype) in ("str", "string"):
            out.isetitem(i, col.map(_clean_text))
    return out


# =====================================================================
# 2. BASIC DATA QUALITY: NULLS, DUPLICATES, COLUMN PROFILE
# =====================================================================
def compute_summary_row(df: pd.DataFrame, file_name: str, sheet_name: str) -> dict:
    rows, cols = df.shape
    total_cells = rows * cols
    total_nulls = int(df.isnull().sum().sum())
    null_pct = round((total_nulls / total_cells) * 100, 2) if total_cells else 0.0
    dup_rows = int(df.duplicated().sum())
    dup_pct = round((dup_rows / rows) * 100, 2) if rows else 0.0

    return {
        "File Name": file_name,
        "Sheet Name": sheet_name,
        "Rows": rows,
        "Columns": cols,
        "Total Null Values": total_nulls,
        "Null Percentage (%)": null_pct,
        "Duplicate Rows": dup_rows,
        "Duplicate Percentage (%)": dup_pct,
    }


def compute_null_details(df: pd.DataFrame, file_name: str, sheet_name: str) -> list:
    rows = len(df)
    details = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            details.append({
                "File Name": file_name,
                "Sheet Name": sheet_name,
                "Column Name": col,
                "Null Count": null_count,
                "Null Percentage (%)": round((null_count / rows) * 100, 2) if rows else 0.0,
            })
    return details


def compute_column_profile(df: pd.DataFrame, sample_size: int = 50) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        rows.append({
            "Column Name": _clean_text(str(col)),
            "Data Type": str(df[col].dtype),
            "Unique Count": len(unique_vals),
            "Null Count": int(df[col].isnull().sum()),
            "Unique Values (sample)": _clean_text(", ".join(map(str, unique_vals[:sample_size]))),
        })
    return pd.DataFrame(rows)


# =====================================================================
# 3. PII DETECTION ENGINE
# =====================================================================
PII_KEYWORDS = {
    "PERSON_NAME": ["name", "first_name", "last_name", "full_name", "customer_name",
                    "patient_name", "employee_name"],
    "EMAIL": ["email", "mail", "email_id"],
    "PHONE": ["phone", "mobile", "contact", "telephone"],
    "AADHAAR": ["aadhaar", "aadhar", "uid"],
    "PAN": ["pan"],
    "PASSPORT": ["passport"],
    "ADDRESS": ["address", "street", "location"],
}

REGEX_PATTERNS = {
    "EMAIL": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
    "PHONE": r"^[6-9]\d{9}$",
    "AADHAAR": r"^\d{12}$",
    "PAN": r"^[A-Z]{5}[0-9]{4}[A-Z]$",
    "PASSPORT": r"^[A-Z][0-9]{7}$",
}


def detect_pii_by_column_name(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for column in df.columns:
        col = str(column).lower().replace(" ", "_")
        for pii_type, keywords in PII_KEYWORDS.items():
            if any(keyword in col for keyword in keywords):
                results.append({"Column": column, "PII Type": pii_type,
                                 "Detection Method": "Column Name", "Matches": None})
                break
    return pd.DataFrame(results)


def detect_pii_by_regex(df: pd.DataFrame, max_sample: int | None = 5000) -> pd.DataFrame:
    """Regex scan is capped at `max_sample` rows per column for performance on huge files."""
    results = []
    for column in df.columns:
        series = df[column].dropna()
        if len(series) == 0:
            continue
        if max_sample is not None and len(series) > max_sample:
            series = series.sample(max_sample, random_state=42)
        str_series = series.astype(str).str.strip()
        if pd.api.types.is_float_dtype(series):
            # A column of 10/12-digit IDs that contains blanks is read as float
            # ("9876543210.0"), which would otherwise never match the patterns.
            str_series = str_series.str.replace(r"\.0$", "", regex=True)
        for pii_type, pattern in REGEX_PATTERNS.items():
            matches = int(str_series.str.fullmatch(pattern).sum())
            if matches > 0:
                results.append({"Column": column, "PII Type": pii_type,
                                 "Detection Method": "Regex", "Matches": matches})
    return pd.DataFrame(results)


def build_pii_report(df: pd.DataFrame, regex_sample_size: int | None = 5000) -> pd.DataFrame:
    col_df = detect_pii_by_column_name(df)
    regex_df = detect_pii_by_regex(df, max_sample=regex_sample_size)
    if col_df.empty and regex_df.empty:
        return pd.DataFrame(columns=["Column", "PII Type", "Detection Method", "Matches"])
    return pd.concat([col_df, regex_df], ignore_index=True).drop_duplicates()


# =====================================================================
# 4. K-ANONYMITY ENGINE
# =====================================================================
QI_KEYWORDS = [
    "age", "gender", "sex", "dob", "birth", "district", "state", "city", "village",
    "zipcode", "zip", "pincode", "postal", "occupation", "education", "income",
    "salary", "marital", "religion", "ethnicity", "facility", "hospital", "year",
]


def detect_quasi_identifiers(df: pd.DataFrame, max_qi: int | None = 10) -> list:
    candidates = []
    for column in df.columns:
        col = str(column).lower().replace(" ", "_")
        if any(keyword in col for keyword in QI_KEYWORDS):
            candidates.append(column)
    if max_qi is not None and len(candidates) > max_qi:
        LOGGER.debug("Detected %d candidate QIs, capping at %d for k-anonymity calc.",
                     len(candidates), max_qi)
        candidates = candidates[:max_qi]
    return candidates


def compute_k_anonymity(df: pd.DataFrame, quasi_identifiers: list, k_threshold: int = 5):
    """Returns (k_value, group_sizes_df, violations_df)."""
    qis_present = [c for c in quasi_identifiers if c in df.columns]
    if not qis_present:
        return None, pd.DataFrame(), pd.DataFrame()

    group_sizes = df.groupby(qis_present, dropna=False).size().reset_index(name="Group_Size")
    if group_sizes.empty:  # empty table: no groups, k is undefined
        return None, group_sizes, pd.DataFrame()
    k_value = int(group_sizes["Group_Size"].min())
    violations = group_sizes[group_sizes["Group_Size"] < k_threshold]
    return k_value, group_sizes, violations


# =====================================================================
# 5. AI KOSH DATA QUALITY SCORING FRAMEWORK  (Max 60)
# =====================================================================
def data_volume_score(df: pd.DataFrame, min_rows=5_000, max_rows=100_000, max_score=10) -> float:
    num_rows = len(df)
    if num_rows < min_rows:
        return 0.0
    score = max_score * (
        (math.log10(num_rows) - math.log10(min_rows)) /
        (math.log10(max_rows) - math.log10(min_rows))
    )
    return round(float(min(score, max_score)), 2)


def missing_value_score(df: pd.DataFrame, max_score=15, threshold=0.20):
    missing_ratio = df.isnull().mean()
    col_scores = [0 if r >= threshold else max_score * (1 - (r / threshold)) for r in missing_ratio]
    dataset_score = round(float(np.mean(col_scores)), 2) if col_scores else 0.0
    return dataset_score, pd.Series(col_scores, index=df.columns)


def schema_consistency_score(df: pd.DataFrame, max_score=5):
    total_columns = len(df.columns)
    if total_columns == 0:
        return 0.0, {}
    inconsistent, report = 0, {}
    for col in df.columns:
        series = df[col].dropna()
        if series.dtype != "object":
            report[col] = "Consistent"
            continue
        unique_types = series.map(type).unique()
        if len(unique_types) > 1:
            inconsistent += 1
            report[col] = "Inconsistent"
        else:
            report[col] = "Consistent"
    score = max_score * (1 - (inconsistent / total_columns))
    return round(float(score), 2), report


def uniqueness_score(df: pd.DataFrame, max_score=5) -> float:
    total_rows = len(df)
    if total_rows == 0:
        return 0.0
    dup_ratio = df.duplicated().sum() / total_rows
    return round(float(max(0.0, max_score * (1 - 10 * dup_ratio))), 2)


def outlier_score(df: pd.DataFrame, max_score=5, max_ratio=0.20) -> float:
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        return 0.0
    scores = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            scores.append(max_score)
            continue
        outliers = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        ratio = outliers.mean()
        scores.append(max(0.0, max_score * (1 - (ratio / max_ratio))))
    return round(float(np.mean(scores)), 2) if scores else 0.0


def _entropy_from_probs(probs) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _choose_bins(series: pd.Series, min_bins: int = 3, max_bins: int = 50) -> int:
    """Freedman-Diaconis rule: adapts to sample size, spread & outliers -
    no hard-coded row-count thresholds."""
    x = series.dropna().values
    if len(x) < 2:
        return min_bins
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        return max(min_bins, min(int(np.sqrt(len(x))), max_bins))
    bin_width = 2 * iqr * (len(x) ** (-1 / 3))
    if bin_width <= 0:
        return min_bins
    data_range = x.max() - x.min()
    if data_range == 0:
        return min_bins
    bins = int(np.ceil(data_range / bin_width))
    return max(min_bins, min(bins, max_bins))


def compute_entropy_score(df: pd.DataFrame, max_score: int = 20) -> float:
    scores = []
    cat_cols = df.select_dtypes(exclude=np.number).columns
    num_cols = df.select_dtypes(include=np.number).columns

    for col in cat_cols:
        series = df[col].dropna()
        if len(series) == 0 or series.nunique() <= 1:
            scores.append(0.0)
            continue
        probs = series.value_counts(normalize=True).values
        scores.append(max_score * _entropy_from_probs(probs))

    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0 or series.nunique() <= 1:
            scores.append(0.0)
            continue
        bins = _choose_bins(series)
        try:
            binned = pd.qcut(series, q=bins, duplicates="drop")
        except Exception:
            binned = pd.cut(series, bins=bins)
        probs = binned.value_counts(normalize=True).values
        scores.append(max_score * _entropy_from_probs(probs))

    return round(float(np.mean(scores)), 2) if scores else 0.0


def data_quality_score(df: pd.DataFrame) -> pd.Series:
    breakdown = {}
    breakdown["Data Volume (10)"] = data_volume_score(df)
    breakdown["Missing Values (15)"], _ = missing_value_score(df)
    breakdown["Schema Consistency (5)"], _ = schema_consistency_score(df)
    breakdown["Uniqueness (5)"] = uniqueness_score(df)
    breakdown["Outliers (5)"] = outlier_score(df)
    breakdown["Entropy / Information (20)"] = compute_entropy_score(df)

    total = round(sum(breakdown.values()), 2)
    if total >= 42:
        decision = "GO"
    elif total >= 36:
        decision = "CONDITIONAL GO"
    else:
        decision = "NO GO"

    breakdown["TOTAL SCORE (Max 60)"] = total
    breakdown["FINAL DECISION"] = decision
    return pd.Series(breakdown)


# =====================================================================
# 6. COMBINED DATASET FITNESS  (new - ties quality + privacy together)
# =====================================================================
def combined_fitness_decision(quality_decision: str, privacy_status: str,
                               min_k, k_threshold: int) -> tuple[str, str]:
    """
    Combines the AI Kosh quality decision with the k-anonymity privacy
    status into one manager-facing verdict.

    Rules:
      - Quality NO GO                                -> NOT FIT (quality alone disqualifies)
      - Quality GO            + Privacy PASS/N/A      -> FIT
      - Quality GO            + Privacy FAIL          -> CONDITIONAL (re-identification risk; needs anonymization)
      - Quality CONDITIONAL GO + Privacy PASS/N/A     -> CONDITIONAL (quality needs review)
      - Quality CONDITIONAL GO + Privacy FAIL         -> NOT FIT (quality issues + privacy risk stack up)
    """
    if quality_decision == "NO GO":
        return "NOT FIT", "Quality score failed the minimum bar (decision: NO GO)."

    if quality_decision == "GO":
        if privacy_status in ("PASS", "Not Applicable"):
            reason = ("Quality met the GO threshold and no privacy risk was found "
                      "(no quasi-identifier columns detected)." if privacy_status == "Not Applicable"
                      else f"Quality met the GO threshold and k-anonymity passed (k={min_k} >= {k_threshold}).")
            return "FIT", reason
        return ("CONDITIONAL",
                f"Quality met the GO threshold, but k-anonymity failed (k={min_k} < {k_threshold}) - "
                "re-identification risk. Needs anonymization/generalization before release.")

    if quality_decision == "CONDITIONAL GO":
        if privacy_status == "FAIL":
            return ("NOT FIT",
                    f"Quality only reached CONDITIONAL GO and k-anonymity also failed "
                    f"(k={min_k} < {k_threshold}) - combined quality and privacy risk.")
        reason = ("Quality only reached CONDITIONAL GO; review missing values/entropy before release "
                  "(no privacy risk found)." if privacy_status == "Not Applicable"
                  else f"Quality only reached CONDITIONAL GO (k-anonymity passed, k={min_k}).")
        return "CONDITIONAL", reason

    return "NOT FIT", f"Unrecognized quality decision: {quality_decision!r}."



# =====================================================================
# 7. CROSS-DATASET SIMILARITY & DUPLICATION
# =====================================================================
# This layer compares the underlying input tables, not the generated
# DQ report workbooks. It is intentionally conservative:
#   - schema similarity is always calculated
#   - exact row/key overlap is calculated only when a usable business key
#     is available
#   - automatic key detection is based on common HMIS-style column names
#   - users can override the key with --cross-keys
#
# The analysis is written to the master workbook under:
#   Cross_Dataset_Similarity
#   Cross_Dataset_Keys
#   Cross_Dataset_Schema
#   Cross_Dataset_Guide
#
# The similarity summary also records comparison grain and confidence so
# different reporting grains (for example state/month versus district/year)
# are not interpreted as directly comparable without context.
#
# For very large datasets, --cross-max-rows can be used to sample the
# same number of rows from each table. The report clearly marks the
# comparison as sampled in that case.


def _normalize_column_name(name) -> str:
    """Normalize a column name for matching across datasets."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _canonical_key_name(name) -> str:
    """Return a normalized key name for output/comparison."""
    return _normalize_column_name(name)


# Common business keys for HMIS / health-indicator style datasets.
# Order matters because the first complete candidate is preferred.
AUTO_CROSS_KEY_CANDIDATES = [
    [
        "fiscal_year", "month", "indicator_head", "indicator",
        "facility_category", "category", "unit"
    ],
    [
        "fiscal_year", "month", "state", "indicator_head", "indicator",
        "facility_category", "category", "unit"
    ],
    [
        "fiscal_year", "state", "district", "indicator_head", "indicator",
        "facility_category", "category", "unit"
    ],
    [
        "fiscal_year", "state", "district", "indicator",
        "facility_category", "category", "unit"
    ],
    [
        "fiscal_year", "state", "district_as_per_source", "indicator",
        "unit"
    ],
    [
        "fiscal_year", "state", "district_as_per_lgd", "indicator",
        "unit"
    ],
    [
        "fiscal_year", "state", "district", "indicator", "unit"
    ],
    [
        "year", "month", "indicator_head", "indicator",
        "facility_category", "category", "unit"
    ],
    [
        "year", "state", "district", "indicator", "unit"
    ],
]


def _normalized_column_map(df: pd.DataFrame) -> dict:
    """Map normalized names to original dataframe column names."""
    mapping = {}
    for col in df.columns:
        normalized = _normalize_column_name(col)
        if normalized not in mapping:
            mapping[normalized] = col
    return mapping


def detect_cross_dataset_keys(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    explicit_keys: list[str] | None = None,
) -> tuple[list, str]:
    """
    Detect a shared business key between two datasets.

    Returns:
        (original-column-pairs, method)

    Each pair is (column_in_A, column_in_B).
    """
    map_a = _normalized_column_map(df_a)
    map_b = _normalized_column_map(df_b)

    if explicit_keys:
        pairs = []
        missing = []
        for key in explicit_keys:
            normalized = _normalize_column_name(key)
            if normalized not in map_a:
                missing.append(f"A:{key}")
            if normalized not in map_b:
                missing.append(f"B:{key}")
            if normalized in map_a and normalized in map_b:
                pairs.append((map_a[normalized], map_b[normalized]))

        if missing:
            return [], "EXPLICIT_KEY_MISSING: " + ", ".join(missing)
        return pairs, "EXPLICIT"

    for candidate in AUTO_CROSS_KEY_CANDIDATES:
        if all(k in map_a and k in map_b for k in candidate):
            pairs = [(map_a[k], map_b[k]) for k in candidate]
            return pairs, "AUTO_HMIS_BUSINESS_KEY"

    # Generic fallback: shared columns with identity-like names.
    identity_priority = [
        "id", "record_id", "code", "indicator", "indicator_code",
        "district", "state", "year", "fiscal_year", "month"
    ]
    shared = [
        k for k in identity_priority
        if k in map_a and k in map_b
    ]

    if len(shared) >= 2:
        pairs = [(map_a[k], map_b[k]) for k in shared]
        return pairs, "AUTO_SHARED_IDENTITY_COLUMNS"

    return [], "NO_USABLE_SHARED_BUSINESS_KEY"


def _stable_value(value) -> str:
    """Convert a value into a deterministic comparable representation."""
    if pd.isna(value):
        return "<NULL>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value).strip()


def _normalized_series(series: pd.Series) -> pd.Series:
    """Fast, deterministic comparable representation for a column."""
    return (
        series.astype("string")
        .fillna("<NULL>")
        .str.strip()
    )


def _key_frame(
    df: pd.DataFrame,
    key_pairs: list[tuple],
    side: str,
) -> pd.DataFrame:
    """
    Build a normalized key dataframe using vectorized operations.

    Duplicate keys are retained initially so the report can distinguish
    key overlap from within-dataset key duplication.
    """
    data = {}
    for left, right in key_pairs:
        col = left if side == "A" else right
        key_name = _canonical_key_name(left)
        data[key_name] = _normalized_series(df[col])

    return pd.DataFrame(data, index=df.index)


def _key_strings(key_df: pd.DataFrame) -> pd.Series:
    if key_df.empty:
        return pd.Series(dtype="object")
    cols = [key_df.iloc[:, i].astype("string") for i in range(key_df.shape[1])]
    out = cols[0]
    for c in cols[1:]:
        out = out + "||" + c
    return out


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    """
    Deterministic full-row hashes.

    Values are converted to trimmed strings first so the behavior remains
    consistent across mixed numeric/object columns while avoiding the
    expensive Python-level cell-by-cell map used previously.
    """
    normalized = df.copy()
    normalized.columns = [
        _normalize_column_name(c) for c in normalized.columns
    ]

    for col in normalized.columns:
        normalized[col] = _normalized_series(normalized[col])

    return pd.util.hash_pandas_object(
        normalized,
        index=False,
    ).astype("uint64")



def _format_grain_from_keys(key_names: list[str]) -> str:
    """Create a human-readable reporting grain from normalized business-key columns."""
    labels = {
        "fiscal_year": "Fiscal Year", "year": "Year", "month": "Month",
        "state": "State", "district": "District",
        "district_as_per_source": "District (Source)",
        "district_as_per_lgd": "District (LGD)",
        "indicator_head": "Indicator Head", "indicator": "Indicator",
        "facility_category": "Facility Category", "category": "Category",
        "unit": "Unit", "indicator_code": "Indicator Code",
    }
    ordered = []
    for key in key_names:
        label = labels.get(key, key.replace("_", " ").title())
        if label not in ordered:
            ordered.append(label)
    return " + ".join(ordered) if ordered else "No business key"


def _comparison_confidence(key_method: str, key_names: list[str]) -> tuple[str, str]:
    """Describe confidence in the comparison basis, not dataset quality."""
    if key_method == "EXPLICIT":
        return "HIGH", "Explicit business key supplied by user."
    if key_method == "AUTO_HMIS_BUSINESS_KEY":
        if len(key_names) >= 3:
            return "HIGH", "Recognized HMIS business key detected automatically."
        return "MEDIUM", "HMIS-style key detected, but the key is limited."
    if key_method == "AUTO_SHARED_IDENTITY_COLUMNS":
        return "LOW", "Fallback identity columns used because no HMIS business key matched."
    if key_method.startswith("EXPLICIT_KEY_MISSING"):
        return "LOW", "Requested business key was incomplete for one or both datasets."
    return "N/A", "No usable business key was available; schema comparison only."


def _schema_similarity(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    cols_a = {_normalize_column_name(c) for c in df_a.columns}
    cols_b = {_normalize_column_name(c) for c in df_b.columns}

    union = cols_a | cols_b
    intersection = cols_a & cols_b

    jaccard = (
        len(intersection) / len(union)
        if union else 1.0
    )

    containment_a = (
        len(intersection) / len(cols_a)
        if cols_a else 0.0
    )
    containment_b = (
        len(intersection) / len(cols_b)
        if cols_b else 0.0
    )

    return {
        "Schema Columns A": len(cols_a),
        "Schema Columns B": len(cols_b),
        "Common Columns": len(intersection),
        "A Only Columns": len(cols_a - cols_b),
        "B Only Columns": len(cols_b - cols_a),
        "Schema Jaccard (%)": round(jaccard * 100, 2),
        "A Contained in B (%)": round(containment_a * 100, 2),
        "B Contained in A (%)": round(containment_b * 100, 2),
    }


def compare_two_datasets(
    name_a: str,
    df_a: pd.DataFrame,
    name_b: str,
    df_b: pd.DataFrame,
    explicit_keys: list[str] | None = None,
    max_rows: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Compare two raw input tables.

    Returns:
        summary_dict, key_details_dataframe
    """
    schema = _schema_similarity(df_a, df_b)

    sampled = max_rows is not None and (
        len(df_a) > max_rows or len(df_b) > max_rows
    )

    if max_rows is not None:
        work_a = df_a.head(max_rows).copy()
        work_b = df_b.head(max_rows).copy()
    else:
        work_a = df_a.copy()
        work_b = df_b.copy()

    key_pairs, key_method = detect_cross_dataset_keys(
        work_a,
        work_b,
        explicit_keys=explicit_keys,
    )

    summary = {
        "Dataset A": name_a,
        "Dataset B": name_b,
        "Rows A": len(df_a),
        "Rows B": len(df_b),
        "Rows Compared A": len(work_a),
        "Rows Compared B": len(work_b),
        "Comparison Scope": "SAMPLED" if sampled else "FULL",
        **schema,
        "Key Method": key_method,
        "Business Key": "",
        "Comparison Grain": "",
        "Comparison Confidence": "N/A",
        "Confidence Basis": "",
        "Unique Keys A": None,
        "Unique Keys B": None,
        "Common Keys": None,
        "A Only Keys": None,
        "B Only Keys": None,
        "Key Overlap vs A (%)": None,
        "Key Overlap vs B (%)": None,
        "Key Overlap Jaccard (%)": None,
        "Duplicate Keys A": None,
        "Duplicate Keys B": None,
        "Exact Row Hash Matches": None,
        "Exact Row Match vs A (%)": None,
        "Interpretation": "",
    }

    if not key_pairs:
        summary["Comparison Grain"] = "Schema only"
        summary["Comparison Confidence"] = "N/A"
        summary["Confidence Basis"] = (
            "No usable shared business key. Key-level duplication/overlap was not computed."
        )
        summary["Interpretation"] = (
            "Schema comparison completed, but no shared business key was "
            "available. Exact cross-dataset key duplication was not computed."
        )
        return summary, pd.DataFrame()

    key_names = [_canonical_key_name(pair[0]) for pair in key_pairs]
    summary["Business Key"] = ", ".join(key_names)
    summary["Comparison Grain"] = _format_grain_from_keys(key_names)
    confidence, confidence_basis = _comparison_confidence(key_method, key_names)
    summary["Comparison Confidence"] = confidence
    summary["Confidence Basis"] = confidence_basis

    key_a_df = _key_frame(work_a, key_pairs, "A")
    key_b_df = _key_frame(work_b, key_pairs, "B")

    key_a = _key_strings(key_a_df)
    key_b = _key_strings(key_b_df)

    set_a = set(key_a)
    set_b = set(key_b)

    unique_a = len(set_a)
    unique_b = len(set_b)
    common = len(set_a & set_b)
    only_a = len(set_a - set_b)
    only_b = len(set_b - set_a)

    overlap_a = common / unique_a * 100 if unique_a else 0.0
    overlap_b = common / unique_b * 100 if unique_b else 0.0
    union = len(set_a | set_b)
    jaccard = common / union * 100 if union else 100.0

    duplicate_keys_a = len(key_a) - unique_a
    duplicate_keys_b = len(key_b) - unique_b

    # Exact row comparison is only meaningful when the two datasets have
    # the same normalized schema.
    common_columns = sorted(
        set(_normalize_column_name(c) for c in work_a.columns)
        & set(_normalize_column_name(c) for c in work_b.columns)
    )

    exact_row_matches = None
    exact_row_match_pct = None

    if common_columns and (
        len(common_columns) == len(work_a.columns)
        and len(common_columns) == len(work_b.columns)
    ):
        # Reorder both frames into the same normalized column order.
        map_a = _normalized_column_map(work_a)
        map_b = _normalized_column_map(work_b)

        aligned_a = work_a[[map_a[c] for c in common_columns]].copy()
        aligned_b = work_b[[map_b[c] for c in common_columns]].copy()

        hash_a = set(_row_hashes(aligned_a))
        hash_b = set(_row_hashes(aligned_b))

        exact_row_matches = len(hash_a & hash_b)
        exact_row_match_pct = (
            exact_row_matches / len(hash_a) * 100
            if hash_a else 0.0
        )

    summary.update({
        "Unique Keys A": unique_a,
        "Unique Keys B": unique_b,
        "Common Keys": common,
        "A Only Keys": only_a,
        "B Only Keys": only_b,
        "Key Overlap vs A (%)": round(overlap_a, 2),
        "Key Overlap vs B (%)": round(overlap_b, 2),
        "Key Overlap Jaccard (%)": round(jaccard, 2),
        "Duplicate Keys A": duplicate_keys_a,
        "Duplicate Keys B": duplicate_keys_b,
        "Exact Row Hash Matches": exact_row_matches,
        "Exact Row Match vs A (%)": (
            round(exact_row_match_pct, 2)
            if exact_row_match_pct is not None else None
        ),
    })

    if common == 0:
        interpretation = "No common business keys found."
    elif overlap_a == 100 and overlap_b == 100:
        interpretation = (
            "The compared key domains are identical. Validate values and "
            "lineage before treating the datasets as duplicates."
        )
    elif overlap_a == 100:
        interpretation = (
            "Dataset A keys are fully contained in Dataset B. A may be a "
            "subset or differently valued extract of B."
        )
    elif overlap_b == 100:
        interpretation = (
            "Dataset B keys are fully contained in Dataset A. B may be a "
            "subset or differently valued extract of A."
        )
    else:
        interpretation = (
            "The datasets share some business keys but also contain "
            "dataset-specific records."
        )

    if exact_row_matches is not None and exact_row_matches > 0:
        interpretation += (
            f" Exact full-row matches found: {exact_row_matches}."
        )

    if summary["Comparison Confidence"] == "LOW":
        interpretation += (
            " Caution: this result uses fallback identity columns and should not "
            "be treated as definitive row-level similarity."
        )
    if summary["Comparison Scope"] == "SAMPLED":
        interpretation += (
            " Caution: the comparison used a row sample rather than the full datasets."
        )

    summary["Interpretation"] = interpretation

    # Key-level detail table. This is deliberately capped to avoid writing
    # millions of rows to Excel.
    detail_limit = 10000
    common_keys = sorted(set_a & set_b)[:detail_limit]
    only_a_keys = sorted(set_a - set_b)[:detail_limit]
    only_b_keys = sorted(set_b - set_a)[:detail_limit]

    details = []
    for key in common_keys:
        details.append({
            "Dataset A": name_a,
            "Dataset B": name_b,
            "Key": key,
            "Presence": "COMMON",
        })
    for key in only_a_keys:
        details.append({
            "Dataset A": name_a,
            "Dataset B": name_b,
            "Key": key,
            "Presence": "A_ONLY",
        })
    for key in only_b_keys:
        details.append({
            "Dataset A": name_a,
            "Dataset B": name_b,
            "Key": key,
            "Presence": "B_ONLY",
        })

    return summary, pd.DataFrame(details)


def _load_comparison_tables(
    input_path: Path,
    recursive: bool,
    max_rows: int | None = None,
) -> list[dict]:
    """
    Load raw input tables for cross-dataset comparison.

    Each Excel sheet becomes a separate comparison table.
    """
    tables = []

    for file_path in discover_files(input_path, recursive):
        try:
            sheets = load_file(file_path, sample_rows=max_rows)
        except Exception as exc:
            LOGGER.warning(
                "Cross-dataset comparison skipped %s: %s",
                file_path.name,
                exc,
            )
            continue

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            # Use the relative dataset path so repeated filenames such as
            # output.csv remain distinguishable in logs and Excel.
            try:
                relative_path = str(file_path.relative_to(input_path))
            except ValueError:
                relative_path = str(file_path)

            relative_path = relative_path.replace("\\", "/")
            table_name = (
                f"{relative_path} | {sheet_name}"
                if sheet_name != "NA"
                else relative_path
            )

            tables.append({
                "name": table_name,
                "file": file_path.name,
                "relative_path": relative_path,
                "sheet": sheet_name,
                "df": df,
            })

    return tables


def run_cross_dataset_analysis(
    input_path: Path,
    recursive: bool = False,
    explicit_keys: list[str] | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run pairwise cross-dataset similarity.

    Returns:
        summary_df, key_details_df, schema_df
    """
    tables = _load_comparison_tables(
        input_path=input_path,
        recursive=recursive,
        max_rows=max_rows,
    )

    if len(tables) < 2:
        LOGGER.info(
            "Cross-dataset similarity skipped: fewer than 2 tables found."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    LOGGER.info(
        "Running cross-dataset similarity across %d input table(s)...",
        len(tables),
    )

    summary_rows = []
    detail_frames = []

    total_pairs = len(tables) * (len(tables) - 1) // 2
    pair_number = 0

    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            pair_number += 1
            a = tables[i]
            b = tables[j]

            LOGGER.info(
                "Cross comparison [%d/%d]: %s <-> %s",
                pair_number,
                total_pairs,
                a["name"],
                b["name"],
            )

            summary, details = compare_two_datasets(
                name_a=a["name"],
                df_a=a["df"],
                name_b=b["name"],
                df_b=b["df"],
                explicit_keys=explicit_keys,
                max_rows=max_rows,
            )

            summary_rows.append(summary)

            if not details.empty:
                detail_frames.append(details)

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        def _similarity_status(row) -> str:
            common = row.get("Common Keys")
            overlap_a = row.get("Key Overlap vs A (%)")
            overlap_b = row.get("Key Overlap vs B (%)")
            exact_rows = row.get("Exact Row Hash Matches")

            if pd.isna(common):
                return "SCHEMA ONLY"

            if common == 0:
                return "NO KEY OVERLAP"

            if (
                pd.notna(overlap_a)
                and pd.notna(overlap_b)
                and overlap_a == 100
                and overlap_b == 100
            ):
                if pd.notna(exact_rows) and exact_rows and exact_rows > 0:
                    return "POTENTIAL DUPLICATE / IDENTICAL KEY DOMAIN"
                return "IDENTICAL KEY DOMAIN"

            if pd.notna(overlap_a) and overlap_a == 100:
                return "A KEYS SUBSET OF B"

            if pd.notna(overlap_b) and overlap_b == 100:
                return "B KEYS SUBSET OF A"

            return "PARTIAL KEY OVERLAP"

        summary_df.insert(
            0,
            "Similarity Status",
            summary_df.apply(_similarity_status, axis=1),
        )

    if detail_frames:
        key_details_df = pd.concat(
            detail_frames,
            ignore_index=True,
        )
    else:
        key_details_df = pd.DataFrame(
            columns=[
                "Dataset A",
                "Dataset B",
                "Key",
                "Presence",
            ]
        )

    # Dataset-level schema inventory is useful even where no business key
    # could be detected.
    schema_rows = []
    for table in tables:
        df = table["df"]
        for col in df.columns:
            schema_rows.append({
                "Dataset": table["name"],
                "Relative Path": table.get("relative_path", table["file"]),
                "File": table["file"],
                "Sheet": table["sheet"],
                "Column": col,
                "Normalized Column": _normalize_column_name(col),
                "Data Type": str(df[col].dtype),
                "Rows": len(df),
                "Unique Values": int(df[col].nunique(dropna=True)),
                "Null Count": int(df[col].isna().sum()),
            })

    schema_df = pd.DataFrame(schema_rows)

    LOGGER.info(
        "Cross-dataset analysis complete: %d pair(s) compared.",
        len(summary_df),
    )

    return summary_df, key_details_df, schema_df



# =====================================================================
# 8. VISUALIZATION  (new)
# =====================================================================
def plot_score_breakdown(score: pd.Series, table_key: str, chart_dir: Path) -> Path | None:
    try:
        components = [
            ("Data Volume", score["Data Volume (10)"], 10),
            ("Missing Values", score["Missing Values (15)"], 15),
            ("Schema Consistency", score["Schema Consistency (5)"], 5),
            ("Uniqueness", score["Uniqueness (5)"], 5),
            ("Outliers", score["Outliers (5)"], 5),
            ("Entropy", score["Entropy / Information (20)"], 20),
        ]
        labels = [c[0] for c in components]
        achieved = [c[1] for c in components]
        maximum = [c[2] for c in components]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(labels))
        ax.bar(x, maximum, color="#E0E0E0", label="Max possible")
        ax.bar(x, achieved, color="#2E86AB", label="Achieved")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"AI Kosh Score Breakdown - {table_key}\n"
                     f"Total: {score['TOTAL SCORE (Max 60)']}/60 ({score['FINAL DECISION']})")
        ax.legend()
        fig.tight_layout()

        chart_dir.mkdir(parents=True, exist_ok=True)
        out_path = chart_dir / f"{table_key}_score_breakdown.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path
    except Exception as exc:
        LOGGER.debug("Chart generation (score breakdown) failed for %s: %s", table_key, exc)
        return None


def plot_missing_values(df: pd.DataFrame, table_key: str, chart_dir: Path, top_n: int = 15) -> Path | None:
    try:
        missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0].head(top_n)
        if missing_pct.empty:
            return None

        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(missing_pct))))
        ax.barh(missing_pct.index[::-1], missing_pct.values[::-1], color="#C1121F")
        ax.set_xlabel("Missing (%)")
        ax.set_title(f"Top Missing-Value Columns - {table_key}")
        fig.tight_layout()

        chart_dir.mkdir(parents=True, exist_ok=True)
        out_path = chart_dir / f"{table_key}_missing_values.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path
    except Exception as exc:
        LOGGER.debug("Chart generation (missing values) failed for %s: %s", table_key, exc)
        return None


def plot_fitness_distribution(master_df: pd.DataFrame, output_dir: Path) -> Path | None:
    try:
        if "DATASET FITNESS" not in master_df.columns:
            return None
        counts = master_df["DATASET FITNESS"].value_counts()
        counts = counts[counts.index.isin(["FIT", "CONDITIONAL", "NOT FIT"])]
        if counts.empty:
            return None

        colors = {"FIT": "#2A9D8F", "CONDITIONAL": "#E9C46A", "NOT FIT": "#E76F51"}
        order = [k for k in ["FIT", "CONDITIONAL", "NOT FIT"] if k in counts.index]
        counts = counts.reindex(order)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(counts.index, counts.values, color=[colors[k] for k in counts.index])
        ax.set_ylabel("Number of tables")
        ax.set_title("Dataset Fitness - Batch Summary")
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(int(v)), ha="center", va="bottom")
        fig.tight_layout()

        chart_dir = output_dir / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        out_path = chart_dir / "fitness_distribution.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return out_path
    except Exception as exc:
        LOGGER.debug("Chart generation (fitness distribution) failed: %s", exc)
        return None


def embed_charts_sheet(writer, sheet_name: str, charts: dict) -> None:
    """Creates a sheet and embeds every chart PNG in `charts` (table_key -> [Path,...])."""
    any_chart = any(paths for paths in charts.values())
    if not any_chart:
        return
    ws = writer.book.create_sheet(sheet_name)
    row_cursor = 1
    for table_key, paths in charts.items():
        real_paths = [p for p in paths if p and Path(p).exists()]
        if not real_paths:
            continue
        ws.cell(row=row_cursor, column=1, value=f"Table: {table_key}")
        row_cursor += 1
        for img_path in real_paths:
            img = XLImage(str(img_path))
            ws.add_image(img, f"A{row_cursor}")
            row_cursor += 22  # approx rows to clear the embedded image


# =====================================================================
# 9. PER-FILE PROCESSING & REPORT WRITER
# =====================================================================
def sanitize_sheet_name(name: str, used: set | None = None) -> str:
    """Excel sheet names: <=31 chars, no []:*?/\\ characters, must be unique per workbook."""
    for ch in "[]:*?/\\":
        name = name.replace(ch, "_")
    name = (name or "Sheet")[:31]
    if used is not None:
        base, suffix = name, 1
        while name in used:
            suffix += 1
            tail = f"_{suffix}"
            name = base[: 31 - len(tail)] + tail
        used.add(name)
    return name


def process_file(file_path: Path, k_threshold: int = 5, sample_size: int = 50,
                  max_rows: int | None = None, max_qi: int | None = 10,
                  regex_sample_size: int | None = 5000, enable_charts: bool = True,
                  chart_dir: Path | None = None) -> dict:
    """Runs the full pipeline on every sheet of a single file."""
    result = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "summary_rows": [],
        "null_details": [],
        "column_profiles": {},
        "pii_reports": {},
        "kanon_groups": {},
        "kanon_violations": {},
        "aikosh_scores": {},
        "charts": {},
        "errors": [],
    }

    try:
        sheets = load_file(file_path, sample_rows=max_rows)
    except Exception as exc:
        LOGGER.error("Failed to read %s: %s", file_path.name, exc)
        result["errors"].append(f"READ ERROR: {exc}")
        return result

    used_table_keys: set = set()
    for sheet_name, df in sheets.items():
        table_key = sanitize_sheet_name(f"{file_path.stem}_{sheet_name}", used_table_keys)

        try:
            result["summary_rows"].append(compute_summary_row(df, file_path.name, sheet_name))
            result["null_details"].extend(compute_null_details(df, file_path.name, sheet_name))
            result["column_profiles"][table_key] = compute_column_profile(df, sample_size)
            result["pii_reports"][table_key] = build_pii_report(df, regex_sample_size=regex_sample_size)

            qis = detect_quasi_identifiers(df, max_qi=max_qi)
            k_val, group_sizes, violations = compute_k_anonymity(df, qis, k_threshold)
            result["kanon_groups"][table_key] = group_sizes
            result["kanon_violations"][table_key] = violations

            score = data_quality_score(df)

            # --- privacy fields ---
            privacy_status = "Not Applicable" if k_val is None else ("PASS" if k_val >= k_threshold else "FAIL")
            # Note: intentionally avoid the literal "N/A" - pandas/Excel-reading tools
            # silently treat that exact string as a missing value on re-import.
            score["Quasi Identifiers"] = ", ".join(qis) if qis else "None detected"
            score["Min K-Value"] = k_val if k_val is not None else "Not Applicable"
            score["K Threshold"] = k_threshold
            score["K (min/threshold)"] = f"{k_val} / {k_threshold}" if k_val is not None else "Not Applicable"
            score["K-Anonymity Status"] = privacy_status

            # --- combined dataset fitness (new) ---
            fitness, reason = combined_fitness_decision(score["FINAL DECISION"], privacy_status, k_val, k_threshold)
            score["DATASET FITNESS"] = fitness
            score["FITNESS REASON"] = reason

            # --- context fields for the executive summary ---
            score["Sheet Name"] = sheet_name
            score["Rows"] = df.shape[0]
            score["Columns"] = df.shape[1]

            result["aikosh_scores"][table_key] = score

            if enable_charts and chart_dir is not None:
                score_chart = plot_score_breakdown(score, table_key, chart_dir)
                missing_chart = plot_missing_values(df, table_key, chart_dir)
                result["charts"][table_key] = [p for p in (score_chart, missing_chart) if p]
            else:
                result["charts"][table_key] = []

        except Exception as exc:
            LOGGER.error("Error processing sheet '%s' in %s: %s", sheet_name, file_path.name, exc)
            result["errors"].append(f"{sheet_name}: {exc}")

    return result


def write_report(result: dict, output_dir: Path) -> Path:
    """Writes one consolidated Excel workbook for a single input file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = Path(result["file_path"])

    # Use the dataset folder + file stem so identical filenames
    # from different datasets never overwrite each other.
    dataset_name = file_path.parent.name
    file_stem = file_path.stem

    safe_dataset = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset_name).strip("_")
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", file_stem).strip("_")

    out_path = output_dir / f"DQ_Report_{safe_dataset}_{safe_stem}.xlsx"

    summary_df = pd.DataFrame(result["summary_rows"])
    null_df = pd.DataFrame(result["null_details"])

    aikosh_rows = []
    for table_key, score in result["aikosh_scores"].items():
        row = {"Table": table_key}
        row.update(score.to_dict())
        aikosh_rows.append(row)
    aikosh_df = pd.DataFrame(aikosh_rows)

    # --- manager-facing Executive Summary (new) ---
    if not aikosh_df.empty:
        exec_df = aikosh_df.rename(columns={
            "FINAL DECISION": "Quality Decision",
            "TOTAL SCORE (Max 60)": "Quality Score (/60)",
            "K-Anonymity Status": "Privacy Status",
        })
        exec_cols = ["DATASET FITNESS", "Table", "Sheet Name", "Rows", "Columns",
                     "Quality Decision", "Quality Score (/60)", "Privacy Status",
                     "K (min/threshold)", "Quasi Identifiers", "FITNESS REASON"]
        exec_df = exec_df[[c for c in exec_cols if c in exec_df.columns]]
    else:
        exec_df = pd.DataFrame({"Info": ["No tables were successfully processed for this file."]})

    used_sheet_names: set = set()

    def sheet(name: str) -> str:
        return sanitize_sheet_name(name, used_sheet_names)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Executive_Summary goes first so it's the sheet a manager opens by default.
        _xl_safe_df(exec_df).to_excel(writer, sheet_name=sheet("Executive_Summary"), index=False)

        (summary_df if not summary_df.empty else pd.DataFrame({"Info": ["No data"]})) \
            .to_excel(writer, sheet_name=sheet("Summary"), index=False)
        (_xl_safe_df(null_df) if not null_df.empty else pd.DataFrame({"Info": ["No nulls found"]})) \
            .to_excel(writer, sheet_name=sheet("Null_Details"), index=False)
        (_xl_safe_df(aikosh_df) if not aikosh_df.empty else pd.DataFrame({"Info": ["No score"]})) \
            .to_excel(writer, sheet_name=sheet("AIKosh_Score"), index=False)

        for table_key, profile_df in result["column_profiles"].items():
            _xl_safe_df(profile_df).to_excel(writer, sheet_name=sheet(f"Profile_{table_key}"), index=False)

        for table_key, pii_df in result["pii_reports"].items():
            (_xl_safe_df(pii_df) if not pii_df.empty else pd.DataFrame({"Info": ["No PII detected"]})) \
                .to_excel(writer, sheet_name=sheet(f"PII_{table_key}"), index=False)

        for table_key, group_df in result["kanon_groups"].items():
            if not group_df.empty:
                _xl_safe_df(group_df, "Group_Size", f"KAnon_{table_key}") \
                    .to_excel(writer, sheet_name=sheet(f"KAnon_{table_key}"), index=False)

        for table_key, viol_df in result["kanon_violations"].items():
            if not viol_df.empty:
                _xl_safe_df(viol_df, "Group_Size", f"KViol_{table_key}") \
                    .to_excel(writer, sheet_name=sheet(f"KViol_{table_key}"), index=False)

        if result["errors"]:
            pd.DataFrame({"Errors": result["errors"]}).to_excel(writer, sheet_name=sheet("Errors"), index=False)

        embed_charts_sheet(writer, sheet("Visualization"), result.get("charts", {}))

    return out_path


# =====================================================================
# 10. FOLDER-LEVEL PIPELINE
# =====================================================================
def discover_files(input_path: Path, recursive: bool = False) -> list:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in input_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def run_pipeline(input_path: str, output_path: str, k_threshold: int = 5,
                  recursive: bool = False, sort_by_decision: bool = True,
                  sample_size: int = 50, max_rows: int | None = None,
                  max_qi: int | None = 10, regex_sample_size: int | None = 5000,
                  enable_charts: bool = True,
                  cross_dataset: bool = True,
                  cross_keys: list[str] | None = None,
                  cross_max_rows: int | None = None,
                  collect: dict | None = None) -> pd.DataFrame:
    """Run the full batch. If `collect` (a dict) is passed it is filled with the
    per-file result dicts and cross-dataset frames so callers such as the web UI
    can reuse them without recomputing. Scoring/decisions are unaffected."""
    if collect is not None:
        collect.update({"results": [], "cross_summary": pd.DataFrame(),
                        "cross_keys": pd.DataFrame(), "cross_schema": pd.DataFrame(),
                        "master_path": None, "failed": []})

    input_path, output_path = Path(input_path), Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_path, recursive)
    if not files:
        LOGGER.warning("No supported files found in %s (looked for %s)",
                        input_path, ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        return pd.DataFrame()

    LOGGER.info("Found %d file(s) to process.", len(files))
    master_rows = []

    for idx, file_path in enumerate(files, start=1):
        LOGGER.info("[%d/%d] Processing %s", idx, len(files), file_path.name)
        safe_dataset = re.sub(r"[^A-Za-z0-9_-]+", "_", file_path.parent.name).strip("_")
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", file_path.stem).strip("_")
        chart_dir = output_path / "charts" / f"{safe_dataset}_{safe_stem}"
        result = process_file(file_path, k_threshold=k_threshold,
                               sample_size=sample_size, max_rows=max_rows,
                               max_qi=max_qi, regex_sample_size=regex_sample_size,
                               enable_charts=enable_charts, chart_dir=chart_dir)

        try:
            folder_label = str(file_path.parent.relative_to(input_path)) if input_path.is_dir() else "."
        except ValueError:
            folder_label = str(file_path.parent)

        if result["errors"] and not result["aikosh_scores"]:
            master_rows.append({"File": file_path.name, "Folder": folder_label, "Table": "N/A",
                                 "DATASET FITNESS": "FAILED", "FINAL DECISION": "FAILED",
                                 "Notes": "; ".join(result["errors"])})
            if collect is not None:
                collect["failed"].append({"file": file_path.name, "errors": result["errors"]})
            continue

        try:
            report_path = write_report(result, output_path / "reports")
        except Exception as exc:  # one unwritable report must not abort the batch
            LOGGER.error("Could not write report for %s: %s", file_path.name, exc)
            result["errors"].append(f"REPORT WRITE ERROR: {exc}")
            report_path = None
        result["report_path"] = str(report_path) if report_path else None
        if collect is not None:
            collect["results"].append(result)

        for table_key, score in result["aikosh_scores"].items():
            row = {"File": file_path.name, "Folder": folder_label, "Table": table_key,
                   "Report": str(report_path) if report_path else "NOT WRITTEN"}
            row.update(score.to_dict())
            master_rows.append(row)

        if sort_by_decision and report_path is not None:
            for score in result["aikosh_scores"].values():
                bucket_name = str(score["DATASET FITNESS"]).replace(" ", "_")
                bucket_dir = output_path / bucket_name
                bucket_dir.mkdir(parents=True, exist_ok=True)
                bucket_report = bucket_dir / report_path.name
                if not bucket_report.exists():
                    shutil.copy2(report_path, bucket_report)

    master_df = pd.DataFrame(master_rows)
    master_out = output_path / "Master_Data_Quality_Report.xlsx"

    # ---------------------------------------------------------------
    # Cross-dataset similarity / duplication
    # ---------------------------------------------------------------
    cross_summary_df = pd.DataFrame()
    cross_keys_df = pd.DataFrame()
    cross_schema_df = pd.DataFrame()

    if cross_dataset:
        try:
            cross_summary_df, cross_keys_df, cross_schema_df = (
                run_cross_dataset_analysis(
                    input_path=input_path,
                    recursive=recursive,
                    explicit_keys=cross_keys,
                    max_rows=cross_max_rows,
                )
            )
        except Exception as exc:
            LOGGER.exception(
                "Cross-dataset similarity analysis failed: %s",
                exc,
            )

    if collect is not None:
        collect.update({"cross_summary": cross_summary_df, "cross_keys": cross_keys_df,
                        "cross_schema": cross_schema_df})

    if not master_df.empty:
        fitness_chart = plot_fitness_distribution(master_df, output_path) if enable_charts else None

        try:
            with pd.ExcelWriter(master_out, engine="openpyxl") as writer:
                exec_cols = ["DATASET FITNESS", "File", "Folder", "Table",
                             "FINAL DECISION", "TOTAL SCORE (Max 60)",
                             "K-Anonymity Status", "K (min/threshold)", "FITNESS REASON"]
                exec_master = master_df[[c for c in exec_cols if c in master_df.columns]] \
                    .rename(columns={"FINAL DECISION": "Quality Decision",
                                      "TOTAL SCORE (Max 60)": "Quality Score (/60)",
                                      "K-Anonymity Status": "Privacy Status"})
                exec_master.to_excel(writer, sheet_name="Executive_Summary", index=False)

                master_df.to_excel(writer, sheet_name="Master_Summary", index=False)

                if "DATASET FITNESS" in master_df.columns:
                    fit_counts = master_df["DATASET FITNESS"].value_counts().reset_index()
                    fit_counts.columns = ["Fitness", "Count"]
                    fit_counts.to_excel(writer, sheet_name="Fitness_Counts", index=False)

                if "FINAL DECISION" in master_df.columns:
                    decision_counts = master_df["FINAL DECISION"].value_counts().reset_index()
                    decision_counts.columns = ["Quality Decision", "Count"]
                    decision_counts.to_excel(writer, sheet_name="Decision_Counts", index=False)

                # Cross-dataset analysis sheets
                comparison_guide = pd.DataFrame([
                    {
                        "Field": "Comparison Grain",
                        "Meaning": (
                            "Business dimensions used to define a comparable record, "
                            "for example Fiscal Year + Month + Indicator + Facility Category + Unit."
                        ),
                    },
                    {
                        "Field": "Comparison Confidence",
                        "Meaning": (
                            "Confidence in the comparison basis, not a dataset quality score. "
                            "HIGH = explicit/recognized business key; MEDIUM = limited HMIS key; "
                            "LOW = fallback identity columns; N/A = schema only."
                        ),
                    },
                    {
                        "Field": "Exact Row Hash Matches",
                        "Meaning": (
                            "Identical full rows after normalized schema alignment. "
                            "A match means identical row content, not merely a shared key."
                        ),
                    },
                    {
                        "Field": "Key Overlap",
                        "Meaning": "Percentage of unique business keys shared by the two datasets.",
                    },
                ])
                comparison_guide.to_excel(
                    writer, sheet_name="Cross_Dataset_Guide", index=False
                )

                if not cross_summary_df.empty:
                    cross_summary_df.to_excel(
                        writer,
                        sheet_name="Cross_Dataset_Similarity",
                        index=False,
                    )
                else:
                    pd.DataFrame({
                        "Info": [
                            "No cross-dataset comparison was generated. "
                            "At least two input tables are required."
                        ]
                    }).to_excel(
                        writer,
                        sheet_name="Cross_Dataset_Similarity",
                        index=False,
                    )

                if not cross_keys_df.empty:
                    _xl_safe_df(cross_keys_df, label="Cross_Dataset_Keys").to_excel(
                        writer,
                        sheet_name="Cross_Dataset_Keys",
                        index=False,
                    )

                if not cross_schema_df.empty:
                    _xl_safe_df(cross_schema_df, label="Cross_Dataset_Schema").to_excel(
                        writer,
                        sheet_name="Cross_Dataset_Schema",
                        index=False,
                    )

                if fitness_chart:
                    embed_charts_sheet(writer, "Visualization", {"Batch": [fitness_chart]})

            LOGGER.info("Master report saved to %s", master_out)
            if collect is not None:
                collect['master_path'] = str(master_out)
        except Exception as exc:
            LOGGER.error('Master report could not be written: %s', exc)

    return master_df


# =====================================================================
# 11. CLI
# =====================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Data Quality + PII + K-Anonymity + AI Kosh Scoring + "
                    "Dataset Fitness Toolkit"
    )
    parser.add_argument("--input", "-i", required=True,
                         help="Input file or folder (csv/xls/xlsx/parquet).")
    parser.add_argument("--output", "-o", required=True,
                         help="Output folder for reports.")
    parser.add_argument("--k-threshold", type=int, default=5,
                         help="Minimum acceptable k for k-anonymity (default: 5).")
    parser.add_argument("--recursive", action="store_true",
                         help="Recurse into subfolders when --input is a folder.")
    parser.add_argument("--no-sort", action="store_true",
                         help="Skip sorting reports into FIT/CONDITIONAL/NOT_FIT folders.")
    parser.add_argument("--sample-size", type=int, default=50,
                         help="Max unique values shown per column in the profile (default: 50).")
    parser.add_argument("--max-rows", type=int, default=None,
                         help="If set, only read the first N rows per file/sheet "
                              "(speeds up huge files).")
    parser.add_argument("--max-qi", type=int, default=10,
                         help="Max number of auto-detected quasi-identifier columns to use "
                              "for k-anonymity (default: 10). Use 0 or negative to disable "
                              "the cap and use every candidate found.")
    parser.add_argument("--pii-sample-size", type=int, default=5000,
                         help="Max rows per column sampled for regex-based PII detection "
                              "(default: 5000). Use 0 or negative to scan every row.")
    parser.add_argument("--no-charts", action="store_true",
                         help="Skip chart generation (faster on very large batches).")

    parser.add_argument(
        "--no-cross-dataset",
        action="store_true",
        help="Skip cross-dataset similarity and duplication analysis.",
    )
    parser.add_argument(
        "--cross-keys",
        default=None,
        help=(
            "Comma-separated business-key columns used for cross-dataset "
            "comparison. Example: fiscal_year,month,indicator_head,"
            "indicator,facility_category,category,unit. "
            "If omitted, the tool auto-detects common HMIS-style keys."
        ),
    )
    parser.add_argument(
        "--cross-max-rows",
        type=int,
        default=None,
        help=(
            "Maximum rows loaded per input table for cross-dataset comparison. "
            "Default: full input tables. Use this only when memory/performance "
            "requires sampling."
        ),
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return parser


def main():
    global LOGGER
    args = build_arg_parser().parse_args()
    LOGGER = setup_logger(args.verbose)

    start = datetime.now()
    LOGGER.info("Starting Data Quality Toolkit run...")

    max_qi = None if args.max_qi <= 0 else args.max_qi
    pii_sample_size = None if args.pii_sample_size <= 0 else args.pii_sample_size

    cross_keys = None
    if args.cross_keys:
        cross_keys = [
            key.strip()
            for key in args.cross_keys.split(",")
            if key.strip()
        ]

    if args.cross_max_rows is not None and args.cross_max_rows <= 0:
        cross_max_rows = None
    else:
        cross_max_rows = args.cross_max_rows

    master_df = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        k_threshold=args.k_threshold,
        recursive=args.recursive,
        sort_by_decision=not args.no_sort,
        sample_size=args.sample_size,
        max_rows=args.max_rows,
        max_qi=max_qi,
        regex_sample_size=pii_sample_size,
        enable_charts=not args.no_charts,
        cross_dataset=not args.no_cross_dataset,
        cross_keys=cross_keys,
        cross_max_rows=cross_max_rows,
    )

    elapsed = (datetime.now() - start).total_seconds()
    LOGGER.info("Done in %.1fs. Tables processed: %d", elapsed, len(master_df))

    if not master_df.empty and "DATASET FITNESS" in master_df.columns:
        LOGGER.info("Dataset Fitness breakdown:\n%s",
                    master_df["DATASET FITNESS"].value_counts().to_string())


if __name__ == "__main__":
    main()
