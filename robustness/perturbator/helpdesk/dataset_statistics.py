"""
Dataset statistics for event logs: events, cases, case length/duration,
and static vs dynamic attribute counts.
"""

import pandas as pd
from typing import Dict, Any


def _is_static_column(
    df: pd.DataFrame,
    case_name: str,
    concept_name: str,
    col: str,
    is_categorical: bool,
) -> bool:
    """
    Check if a column is static (same value for all events in each case).

    EOS rows and NaN values are excluded when counting unique values;
    they do not count as distinct values that would make a column dynamic.
    """
    if col not in df.columns:
        return False
    df_no_eos = df[df[concept_name] != "EOS"].copy()
    if df_no_eos.empty:
        return False

    def nunique_excluding_sentinels(series: pd.Series) -> int:
        cleaned = series.dropna()
        if is_categorical:
            cleaned = cleaned[cleaned.astype(str) != "EOS"]
        return cleaned.nunique()

    nunique_per_case = (
        df_no_eos.groupby(case_name)[col]
        .apply(nunique_excluding_sentinels)
    )
    return (nunique_per_case <= 1).all()


def _classify_attributes(
    df: pd.DataFrame, properties: Dict[str, Any]
) -> tuple[int, int, int, int, list[str], list[str]]:
    """Classify attribute columns as static or dynamic. Returns (n_static_cat, n_static_num, n_dynamic_cat, n_dynamic_num, dynamic_cat_names, dynamic_num_names)."""
    case_name = properties["case_name"]
    concept_name = properties["concept_name"]
    cat_cols = properties.get("categorical_columns", [])
    cont_cols = properties.get("continuous_columns", []) + properties.get(
        "continuous_positive_columns", []
    )

    n_static_cat = 0
    dynamic_cat_names: list[str] = []
    for col in cat_cols:
        if _is_static_column(df, case_name, concept_name, col, is_categorical=True):
            n_static_cat += 1
        else:
            dynamic_cat_names.append(col)

    n_static_num = 0
    dynamic_num_names: list[str] = []
    for col in cont_cols:
        if _is_static_column(df, case_name, concept_name, col, is_categorical=False):
            n_static_num += 1
        else:
            dynamic_num_names.append(col)

    return (
        n_static_cat,
        n_static_num,
        len(dynamic_cat_names),
        len(dynamic_num_names),
        dynamic_cat_names,
        dynamic_num_names,
    )


def compute_dataset_statistics(csv_path: str, properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute event log statistics from a CSV file and properties dict.

    Args:
        csv_path: Path to the event log CSV.
        properties: Dict with case_name, concept_name, timestamp_name, date_format,
            categorical_columns, continuous_columns, continuous_positive_columns,
            and optional time column names, min_suffix_size.

    Returns:
        Dict with n_events, n_cases, avg_case_length, avg_case_duration_seconds,
        n_static_categorical, n_static_numerical, n_dynamic_categorical, n_dynamic_numerical,
        dynamic_categorical_names, dynamic_numerical_names.
    """
    from event_log_loader_service.event_log_loader import CSV2EventLog

    event_log = CSV2EventLog(csv_path, **properties)
    df = event_log.df
    case_name = properties["case_name"]
    concept_name = properties["concept_name"]
    timestamp_name = properties["timestamp_name"]

    n_events = len(df)
    n_cases = df[case_name].nunique()
    case_lengths = df.groupby(case_name).size()
    avg_case_length = float(case_lengths.mean())

    df_no_eos = df[df[concept_name] != "EOS"]
    if df_no_eos.empty:
        avg_case_duration_seconds = 0.0
    else:
        durations = (
            df_no_eos.groupby(case_name)[timestamp_name]
            .apply(lambda g: (g.max() - g.min()).total_seconds())
            .dropna()
        )
        avg_case_duration_seconds = float(durations.mean()) if len(durations) > 0 else 0.0

    (
        n_static_cat,
        n_static_num,
        n_dynamic_cat,
        n_dynamic_num,
        dynamic_cat_names,
        dynamic_num_names,
    ) = _classify_attributes(df, properties)

    return {
        "n_events": n_events,
        "n_cases": n_cases,
        "avg_case_length": avg_case_length,
        "avg_case_duration_seconds": avg_case_duration_seconds,
        "n_static_categorical": n_static_cat,
        "n_static_numerical": n_static_num,
        "n_dynamic_categorical": n_dynamic_cat,
        "n_dynamic_numerical": n_dynamic_num,
        "dynamic_categorical_names": dynamic_cat_names,
        "dynamic_numerical_names": dynamic_num_names,
    }
