from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_LANDSCAPE_PARAMS = {
    "Mode1": {"alpha": 0.12, "k": 3},
    "Mode2": {"alpha": 0.12, "k": 3},
    "Mode3": {"alpha": 0.06, "k": 10},
}
DEFAULT_TAU = 0.0
DEFAULT_POWER = 0.4
DEFAULT_LAMBDA_MULTIPLIER = 0.25
DEFAULT_LABEL = "SPES-C"


@dataclass
class SPESParameters:
    mode_hint: Optional[str]
    alpha: float
    k: int
    tau: float = DEFAULT_TAU
    power: float = DEFAULT_POWER
    lambda_multiplier: float = DEFAULT_LAMBDA_MULTIPLIER
    label: str = DEFAULT_LABEL


def _finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if not math.isnan(value) and not math.isinf(value)]


def _mean(values: Iterable[float]) -> float:
    vals = _finite(values)
    return sum(vals) / max(len(vals), 1)


def _std(values: Sequence[float]) -> float:
    vals = _finite(values)
    if len(vals) <= 1:
        return 0.0
    mu = _mean(vals)
    return math.sqrt(sum((value - mu) ** 2 for value in vals) / len(vals))


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _robust_range(values: Sequence[float]) -> Tuple[float, float, float]:
    q05 = _quantile(values, 0.05)
    q95 = _quantile(values, 0.95)
    span = q95 - q05
    if span <= 1e-12:
        finite = _finite(values)
        raw_span = (max(finite) - min(finite)) if finite else 0.0
        span = max(raw_span, _std(finite), 1.0)
    return q05, q95, span


def _normalize_clipped(raw: Sequence[float], reference_q90: float) -> List[float]:
    denom = max(reference_q90, 1e-12)
    return [min(1.0, max(0.0, value / denom)) for value in raw]


def _kernel_weights(x_train: Sequence[float], x_query: float, h: float) -> List[float]:
    weights = [math.exp(-0.5 * ((x - x_query) / h) ** 2) for x in x_train]
    if sum(weights) <= 1e-12 and x_train:
        idx = min(range(len(x_train)), key=lambda i: abs(x_train[i] - x_query))
        weights[idx] = 1.0
    return weights


def _smooth_mean(x_train: Sequence[float], y_train: Sequence[float], x_query: float, h: float) -> float:
    weights = _kernel_weights(x_train, x_query, h)
    num = sum(weight * value for weight, value in zip(weights, y_train))
    den = sum(weights)
    return num / max(den, 1e-12)


def _smooth_mean_vector(
    x_train: Sequence[float],
    y_train: Sequence[float],
    x_query: Sequence[float],
    h: float,
) -> List[float]:
    return [_smooth_mean(x_train, y_train, value, h) for value in x_query]


def _smooth_abs_derivative(
    x_train: Sequence[float],
    y_train: Sequence[float],
    x_query: Sequence[float],
    h: float,
    robust_span: float,
) -> List[float]:
    delta = max(h / 4.0, robust_span * 0.01, 1e-6)
    out: List[float] = []
    for xq in x_query:
        plus = _smooth_mean(x_train, y_train, xq + delta, h)
        minus = _smooth_mean(x_train, y_train, xq - delta, h)
        out.append(abs((plus - minus) / (2.0 * delta)))
    return out


def _knn_mean_distance_train(x_train: Sequence[float], k: int) -> List[float]:
    n = len(x_train)
    if n <= 1:
        return [0.0] * n
    k_eff = min(k, n - 1)
    out = []
    for xq in x_train:
        distances = sorted(abs(x - xq) for x in x_train)
        out.append(sum(distances[1 : k_eff + 1]) / max(k_eff, 1))
    return out


def _knn_mean_distance_query(x_train: Sequence[float], x_query: Sequence[float], k: int) -> List[float]:
    k_eff = min(k, len(x_train))
    out = []
    for xq in x_query:
        distances = sorted(abs(x - xq) for x in x_train)
        out.append(sum(distances[:k_eff]) / max(k_eff, 1))
    return out


def _local_iqr_from_neighbors(
    x_train: Sequence[float],
    residuals: Sequence[float],
    x_query: Sequence[float],
    h: float,
    min_neighbors: int,
) -> List[float]:
    xr = sorted(zip(x_train, residuals), key=lambda item: item[0])
    x_sorted = [item[0] for item in xr]
    r_sorted = [item[1] for item in xr]
    out = []
    for xq in x_query:
        vals = [rv for xv, rv in zip(x_sorted, r_sorted) if (xq - h) <= xv <= (xq + h)]
        if len(vals) < min_neighbors:
            nearest_idx = sorted(range(len(x_sorted)), key=lambda i: abs(x_sorted[i] - xq))[:min_neighbors]
            vals = [r_sorted[i] for i in nearest_idx]
        q25 = _quantile(vals, 0.25)
        q75 = _quantile(vals, 0.75)
        out.append(q75 - q25)
    return out


def _ecdf_percent(train_reference: Sequence[float], values: Sequence[float]) -> List[float]:
    ref = sorted(_finite(train_reference))
    n = len(ref)
    out = []
    for value in values:
        rank = 0
        for ref_value in ref:
            if ref_value <= value:
                rank += 1
            else:
                break
        out.append(100.0 * rank / max(n, 1))
    return out


def _soft_or4(s: float, d: float, b: float, h: float) -> float:
    s = min(1.0, max(0.0, s))
    d = min(1.0, max(0.0, d))
    b = min(1.0, max(0.0, b))
    h = min(1.0, max(0.0, h))
    return 1.0 - (1.0 - s) * (1.0 - d) * (1.0 - b) * (1.0 - h)


def _build_feature_landscape(
    x_train: Sequence[float],
    shap_train: Sequence[float],
    x_query: Sequence[float],
    alpha: float,
    k: int,
) -> Dict[str, List[float] | float]:
    q05, q95, robust_span = _robust_range(x_train)
    h = max(alpha * robust_span, 1e-6)

    trend_train = _smooth_mean_vector(x_train, shap_train, x_train, h)
    s_train_raw = _smooth_abs_derivative(x_train, shap_train, x_train, h, robust_span)
    s_query_raw = _smooth_abs_derivative(x_train, shap_train, x_query, h, robust_span)
    s_ref = _quantile(s_train_raw, 0.90)
    s_train = _normalize_clipped(s_train_raw, s_ref)
    s_query = _normalize_clipped(s_query_raw, s_ref)

    d_train_raw = _knn_mean_distance_train(x_train, k)
    d_query_raw = _knn_mean_distance_query(x_train, x_query, k)
    d_q50 = _quantile(d_train_raw, 0.50)
    d_q90 = _quantile(d_train_raw, 0.90)
    d_scale = max(d_q90 - d_q50, 1e-12)
    d_train = [min(1.0, max(0.0, (value - d_q50) / d_scale)) for value in d_train_raw]
    d_query = [min(1.0, max(0.0, (value - d_q50) / d_scale)) for value in d_query_raw]

    bd_train = [min(abs(x - q05), abs(q95 - x)) for x in x_train]
    bd_query = [min(abs(x - q05), abs(q95 - x)) for x in x_query]
    b_train = [math.exp(-value / h) for value in bd_train]
    b_query = [math.exp(-value / h) for value in bd_query]

    resid_train = [sv - tv for sv, tv in zip(shap_train, trend_train)]
    min_neighbors = max(5, min(len(x_train), k * 4))
    h_train_raw = _local_iqr_from_neighbors(x_train, resid_train, x_train, h, min_neighbors=min_neighbors)
    h_query_raw = _local_iqr_from_neighbors(x_train, resid_train, x_query, h, min_neighbors=min_neighbors)
    h_ref = _quantile(h_train_raw, 0.90)
    h_train = _normalize_clipped(h_train_raw, h_ref)
    h_query = _normalize_clipped(h_query_raw, h_ref)

    return {
        "s_train": s_train,
        "s_query": s_query,
        "d_train": d_train,
        "d_query": d_query,
        "b_train": b_train,
        "b_query": b_query,
        "h_train": h_train,
        "h_query": h_query,
    }


def infer_mode_hint(*texts: object) -> Optional[str]:
    for text in texts:
        token = str(text or "").upper()
        if "MODE3" in token:
            return "Mode3"
        if "MODE2" in token:
            return "Mode2"
        if "MODE1" in token:
            return "Mode1"
    return None


def resolve_spes_parameters(
    mode_hint: Optional[str] = None,
    alpha: Optional[float] = None,
    k: Optional[int] = None,
    tau: float = DEFAULT_TAU,
    power: float = DEFAULT_POWER,
    lambda_multiplier: float = DEFAULT_LAMBDA_MULTIPLIER,
    label: str = DEFAULT_LABEL,
) -> SPESParameters:
    mode_key = mode_hint if mode_hint in DEFAULT_LANDSCAPE_PARAMS else None
    defaults = DEFAULT_LANDSCAPE_PARAMS.get(mode_key or "Mode1", DEFAULT_LANDSCAPE_PARAMS["Mode1"])
    return SPESParameters(
        mode_hint=mode_key,
        alpha=float(alpha if alpha is not None else defaults["alpha"]),
        k=int(k if k is not None else defaults["k"]),
        tau=float(tau),
        power=float(power),
        lambda_multiplier=float(lambda_multiplier),
        label=label,
    )


def _numeric_series(frame: pd.DataFrame, column: str) -> List[float]:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).tolist()


def build_spes_overlay(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: SPESParameters,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    feature_columns = [col for col in training_df.columns if col.startswith("Feature_")]
    if not feature_columns:
        raise ValueError("Training dataframe does not contain Feature_* columns.")

    shap_columns = [f"SHAP_{col.replace('Feature_', '')}" for col in feature_columns]
    missing_shap = [col for col in shap_columns if col not in training_df.columns]
    if missing_shap:
        raise ValueError(f"Training dataframe is missing SHAP columns: {missing_shap}")

    missing_test_cols = [col for col in feature_columns + shap_columns if col not in test_df.columns]
    if missing_test_cols:
        raise ValueError(f"Test dataframe is missing columns required for SPES: {missing_test_cols}")

    y_train = _numeric_series(training_df, "Target")
    pred_train = _numeric_series(training_df, "Target")
    pred_test_col = "Realtest_Pred" if "Realtest_Pred" in test_df.columns else "Predicted"
    pred_test = _numeric_series(test_df, pred_test_col if pred_test_col in test_df.columns else "Target")

    feature_names = [col.replace("Feature_", "") for col in feature_columns]
    shap_means = []
    for feat_name in feature_names:
        shap_vals = _numeric_series(training_df, f"SHAP_{feat_name}")
        shap_means.append(_mean(abs(value) for value in shap_vals))
    total_shap = sum(shap_means)
    if total_shap <= 1e-12:
        feature_weights = [1.0 / max(len(feature_names), 1)] * len(feature_names)
    else:
        feature_weights = [value / total_shap for value in shap_means]

    p_train_raw = [0.0] * len(training_df)
    p_test_raw = [0.0] * len(test_df)

    for feat_name, weight in zip(feature_names, feature_weights):
        x_train = _numeric_series(training_df, f"Feature_{feat_name}")
        shap_train = _numeric_series(training_df, f"SHAP_{feat_name}")
        x_test = _numeric_series(test_df, f"Feature_{feat_name}")
        landscape = _build_feature_landscape(x_train, shap_train, x_test, params.alpha, params.k)

        a_train = [
            _soft_or4(s, d, b, h)
            for s, d, b, h in zip(
                landscape["s_train"],
                landscape["d_train"],
                landscape["b_train"],
                landscape["h_train"],
            )
        ]
        a_test = [
            _soft_or4(s, d, b, h)
            for s, d, b, h in zip(
                landscape["s_query"],
                landscape["d_query"],
                landscape["b_query"],
                landscape["h_query"],
            )
        ]
        p_train_raw = [base + weight * value for base, value in zip(p_train_raw, a_train)]
        p_test_raw = [base + weight * value for base, value in zip(p_test_raw, a_test)]

    spes_train = _ecdf_percent(p_train_raw, p_train_raw)
    spes_test = _ecdf_percent(p_train_raw, p_test_raw)

    y_scale = _quantile(y_train, 0.90) - _quantile(y_train, 0.50)
    if y_scale <= 1e-12:
        y_scale = max(_std(y_train), 1.0)
    lambda_value = params.lambda_multiplier * y_scale
    denom = max(100.0 - params.tau, 1e-12)
    spes_score = [
        pred + lambda_value * (max(0.0, (min(100.0, max(0.0, pct)) - params.tau) / denom) ** params.power)
        for pred, pct in zip(pred_test, spes_test)
    ]

    overlay_df = test_df.copy()
    overlay_df["SPES_Source_Label"] = params.label
    overlay_df["SPES_Mode"] = params.mode_hint or "Generic"
    overlay_df["SPES_Percentile"] = spes_test
    overlay_df["SPES_Score"] = spes_score
    overlay_df["SPES_Delta"] = [score - pred for score, pred in zip(spes_score, pred_test)]
    overlay_df["SPES_Rank"] = (
        pd.Series(spes_score, index=overlay_df.index).rank(method="first", ascending=False).astype(int)
    )
    overlay_df["SPES_Tau"] = params.tau
    overlay_df["SPES_Power"] = params.power
    overlay_df["SPES_Lambda"] = lambda_value
    overlay_df["SPES_Lambda_Multiplier"] = params.lambda_multiplier
    overlay_df["SPES_Landscape_Alpha"] = params.alpha
    overlay_df["SPES_Landscape_K"] = params.k

    metadata = {
        "label": params.label,
        "mode_hint": params.mode_hint or "Generic",
        "parameters": asdict(params),
        "n_train": len(training_df),
        "n_test": len(test_df),
        "lambda_value": lambda_value,
        "y_scale": y_scale,
        "feature_weights": {
            feat_name: weight for feat_name, weight in zip(feature_names, feature_weights)
        },
        "score_columns": {
            "percentile": "SPES_Percentile",
            "score": "SPES_Score",
            "delta": "SPES_Delta",
            "rank": "SPES_Rank",
        },
        "prediction_column": pred_test_col,
    }
    return overlay_df, metadata


def write_spes_artifacts(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str,
    base_name: str,
    mode_hint: Optional[str] = None,
    alpha: Optional[float] = None,
    k: Optional[int] = None,
    tau: float = DEFAULT_TAU,
    power: float = DEFAULT_POWER,
    lambda_multiplier: float = DEFAULT_LAMBDA_MULTIPLIER,
    label: str = DEFAULT_LABEL,
) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = resolve_spes_parameters(
        mode_hint=mode_hint,
        alpha=alpha,
        k=k,
        tau=tau,
        power=power,
        lambda_multiplier=lambda_multiplier,
        label=label,
    )
    overlay_df, metadata = build_spes_overlay(training_df, test_df, params)

    csv_path = out_dir / f"SPES_Test_Set_Detailed_{base_name}.csv"
    json_path = out_dir / f"SPES_Metadata_{base_name}.json"
    overlay_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}
