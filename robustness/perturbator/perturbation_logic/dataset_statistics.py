"""
Dataset statistics for event logs: events, cases, case length/duration,
and static vs dynamic attribute counts.
"""

import pandas as pd
from typing import Any, Dict

DYNAMIC_THRESHOLD_PCT = 5.0


def _classify_attributes(
    df: pd.DataFrame,
    properties: Dict[str, Any],
    threshold_pct: float = DYNAMIC_THRESHOLD_PCT,
) -> tuple:
    """
    Classify attribute columns as static or dynamic using a per-case "varies"
    check. A feature is dynamic if more than `threshold_pct`% of cases show
    at least one value change within the case.

    NaN values and EOS sentinel values (for categorical columns) are excluded
    from the uniqueness check so they do not artificially inflate variation.

    Returns:
        (n_static_cat, n_static_num, n_dynamic_cat, n_dynamic_num,
         dynamic_cat_names, dynamic_num_names,
         dynamic_details, static_details)

        dynamic_details: list of (col, cases_with_change, pct, is_categorical)
        static_details:  list of (col, is_categorical)
    """
    case_col = properties["case_name"]
    cat_cols = set(properties.get("categorical_columns", []))
    # preserve declared order
    ordered_cols = [
        c
        for c in list(properties.get("categorical_columns", []))
        + list(properties.get("continuous_columns", []))
        + list(properties.get("continuous_positive_columns", []))
        if c in df.columns
    ]

    total_cases = df[case_col].nunique()

    def varies_in_case(x: pd.Series, is_cat: bool) -> bool:
        valid = x.dropna()
        if is_cat:
            valid = valid[valid.astype(str) != "EOS"]
        return valid.nunique() > 1 if len(valid) > 0 else False

    dynamic_details: list = []
    static_details: list = []

    for col in ordered_cols:
        is_cat = col in cat_cols
        cases_with_change = int(
            df.groupby(case_col)[col]
            .apply(lambda x, ic=is_cat: varies_in_case(x, ic))
            .sum()
        )
        pct = 100.0 * cases_with_change / total_cases if total_cases > 0 else 0.0

        if pct > threshold_pct:
            dynamic_details.append((col, cases_with_change, pct, is_cat))
        else:
            static_details.append((col, is_cat))

    dynamic_cat_names = [col for col, _, _, is_cat in dynamic_details if is_cat]
    dynamic_num_names = [col for col, _, _, is_cat in dynamic_details if not is_cat]
    static_cat_names = [col for col, is_cat in static_details if is_cat]
    static_num_names = [col for col, is_cat in static_details if not is_cat]

    return (
        len(static_cat_names),
        len(static_num_names),
        len(dynamic_cat_names),
        len(dynamic_num_names),
        dynamic_cat_names,
        dynamic_num_names,
        dynamic_details,
        static_details,
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
    n_activities = int(df_no_eos[concept_name].nunique())

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
        dynamic_details,
        static_details,
    ) = _classify_attributes(df, properties)

    return {
        "n_events": n_events,
        "n_cases": n_cases,
        "n_activities": n_activities,
        "avg_case_length": avg_case_length,
        "avg_case_duration_seconds": avg_case_duration_seconds,
        "n_static_categorical": n_static_cat,
        "n_static_numerical": n_static_num,
        "n_dynamic_categorical": n_dynamic_cat,
        "n_dynamic_numerical": n_dynamic_num,
        "dynamic_categorical_names": dynamic_cat_names,
        "dynamic_numerical_names": dynamic_num_names,
        # detailed per-attribute results for verbose reporting
        "dynamic_details": dynamic_details,   # list of (col, n_cases, pct, is_cat)
        "static_details": static_details,     # list of (col, is_cat)
    }
