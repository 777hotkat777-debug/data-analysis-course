"""Временной ряд лабораторной работы 3: ряд по периодам, сглаживание, индексы, наивный прогноз."""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 3
ALPHA_EWM = 0.3
BREAK_W = 2
BASE_PERIODS = 4
K_MEAN = 4


def build_series(long: pd.DataFrame, date_col: str, value_col: str, how: str = "median") -> pd.DataFrame:
    """Ряд по периодам: число записей и значение; пустой период остаётся с нулём записей."""
    years = np.sort(long[date_col].unique()).astype("int64")
    step = int(np.gcd.reduce(np.diff(years))) if len(years) > 1 else 1
    grid = pd.Index(range(int(years.min()), int(years.max()) + 1, step), name=date_col)
    grouped = long.groupby(date_col)[value_col]
    out = pd.DataFrame({"records": grouped.size(), "value": getattr(grouped, how)()}).reindex(grid)
    out["records"] = out["records"].fillna(0).astype(int)
    return out


def smooth(series: pd.Series, window: int = WINDOW, alpha: float = ALPHA_EWM) -> pd.DataFrame:
    """Исходный ряд, скользящие среднее и медиана по окну и экспоненциальное сглаживание."""
    return pd.DataFrame(
        {
            "series": series,
            "rolling_mean": series.rolling(window, center=True).mean(),
            "rolling_median": series.rolling(window, center=True).median(),
            "ewm": series.ewm(alpha=alpha).mean(),
        }
    )


def can_decompose(n_periods: int, season) -> bool:
    """Декомпозиция допустима, если задан период сезонности и ряд покрывает два его цикла."""
    return season is not None and season >= 2 and n_periods >= 2 * season


def trend_residual(series: pd.Series, window: int = WINDOW) -> pd.Series:
    """Остаток после вычитания тренда (скользящее среднее); сезонности в ряду нет."""
    return series - series.rolling(window, center=True).mean()


def residual_outliers(resid: pd.Series) -> pd.Series:
    """Периоды, у которых остаток дальше трёх робастных отклонений от медианы остатков."""
    resid = resid.dropna()
    spread = 1.4826 * (resid - resid.median()).abs().median()
    if spread == 0:
        return resid.iloc[0:0]
    return resid[(resid - resid.median()).abs() > 3 * spread]


def break_candidates(series: pd.Series, w: int = BREAK_W) -> pd.DataFrame:
    """Среднее за w периодов до периода и за w после него; наибольшая разность даёт кандидата."""
    values = series.to_numpy(dtype="float64")
    rows = []
    for i in range(w, len(values) - w):
        before = values[i - w:i].mean()
        after = values[i + 1:i + 1 + w].mean()
        rows.append(
            {
                "period": series.index[i],
                "before": float(before),
                "after": float(after),
                "diff": float(after - before),
                "rel_pct": float(100 * (after - before) / before),
            }
        )
    return pd.DataFrame(rows).sort_values("diff", ascending=False).reset_index(drop=True)


def autocorr_table(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """Автокорреляция разностей соседних периодов (ряд без тренда) на лагах 1..max_lag."""
    base = series.diff().dropna()
    rows = [{"lag": lag, "acf": float(base.autocorr(lag=lag))} for lag in range(1, max_lag + 1)]
    out = pd.DataFrame(rows)
    out["bound"] = float(2 / np.sqrt(len(base)))
    out["n_diff"] = int(len(base))
    return out


def to_index(series: pd.Series, base_periods: int = BASE_PERIODS) -> pd.Series:
    """Индекс: значение периода делят на среднее первых base_periods периодов и умножают на 100."""
    return series / series.iloc[:base_periods].mean() * 100


def group_indices(long: pd.DataFrame, date_col: str, value_col: str, group_col: str, groups: list) -> pd.DataFrame:
    """Индексы медианы цели по периодам для выбранных групп."""
    labels = long[group_col].astype("object")
    out = {}
    for name in groups:
        part = long[labels == name]
        out[name] = to_index(part.groupby(date_col)[value_col].median().sort_index())
    return pd.DataFrame(out)


def naive_forecasts(series: pd.Series, h: int, k: int = K_MEAN, season=None):
    """Отрезает хвост из h периодов и строит наивные прогнозы только по началу ряда.

    last: последнее значение; mean_k: среднее за k периодов; drift: продолжение среднего
    прироста обучающей части; seasonal: значение того же периода цикл назад (если есть сезон).
    """
    train, tail = series.iloc[:-h], series.iloc[-h:]
    step = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    forecasts = {
        "last": np.repeat(train.iloc[-1], h),
        "mean_k": np.repeat(train.iloc[-k:].mean(), h),
        "drift": train.iloc[-1] + step * np.arange(1, h + 1),
    }
    if season is not None and season >= 2 and len(train) >= season:
        forecasts["seasonal"] = train.iloc[-season:].to_numpy()[:h]
    return train, tail, forecasts


def mae(fact, pred) -> float:
    return float(np.mean(np.abs(np.asarray(fact, dtype="float64") - np.asarray(pred, dtype="float64"))))


def mape(fact, pred):
    """Средняя относительная ошибка в процентах; при нулях в хвосте не считается."""
    fact = np.asarray(fact, dtype="float64")
    if np.any(fact == 0):
        return None
    return float(np.mean(np.abs(fact - np.asarray(pred, dtype="float64")) / np.abs(fact)) * 100)


def error_table(tail: pd.Series, forecasts: dict) -> pd.DataFrame:
    """MAE и MAPE каждого правила на отложенном хвосте."""
    rows = [{"rule": name, "mae": mae(tail, pred), "mape": mape(tail, pred)} for name, pred in forecasts.items()]
    return pd.DataFrame(rows)
