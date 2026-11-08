"""Повторно используемые расчёты лабораторной работы 2 (разведочный анализ)."""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_STEM = "gh_pop_tot"
TARGET = "pop_tot"
DATE = "year"
GROUP = "country"
GROUP_SRC = "gc_cnt_gad_2025"
ID_COL = "id_uc_g0"
RARE_SHARE = 0.01
OTHER_LABEL = "прочие"


def read_clean(path) -> pd.DataFrame:
    """Читает clean.parquet; файл не изменяется."""
    return pd.read_parquet(Path(path))


def to_long(df: pd.DataFrame, stem: str = TARGET_STEM) -> pd.DataFrame:
    """Переводит многолетние столбцы stem_<год> из широкого вида в длинный."""
    pattern = re.compile(rf"^{re.escape(stem)}_((?:19|20)\d{{2}})$")
    year_cols = {}
    for name in df.columns:
        match = pattern.match(name)
        if match:
            year_cols[name] = int(match.group(1))
    if not year_cols:
        raise ValueError(f"Нет столбцов вида {stem}_<год>")
    keep = [ID_COL, GROUP_SRC]
    absent = [name for name in keep if name not in df.columns]
    if absent:
        raise KeyError(f"В таблице нет столбцов: {absent}")
    wide = df[keep + list(year_cols)]
    long = wide.melt(
        id_vars=keep,
        value_vars=list(year_cols),
        var_name="source_column",
        value_name=TARGET,
    )
    long[DATE] = long["source_column"].map(year_cols).astype("int16")
    long = long.rename(columns={GROUP_SRC: GROUP}).drop(columns="source_column")
    long = long[[ID_COL, GROUP, DATE, TARGET]]
    return long.sort_values([ID_COL, DATE]).reset_index(drop=True)


def period_table(long: pd.DataFrame) -> pd.DataFrame:
    """Ряд по периодам; пустые периоды остаются с нулём записей."""
    years = np.sort(long[DATE].unique())
    step = int(np.gcd.reduce(np.diff(years))) if len(years) > 1 else 1
    grid = pd.Index(range(int(years.min()), int(years.max()) + 1, step), name=DATE)
    grouped = long.groupby(DATE, observed=True)[TARGET]
    out = pd.DataFrame(
        {"n": grouped.size(), "median": grouped.median(), "mean": grouped.mean()}
    ).reindex(grid)
    out["n"] = out["n"].fillna(0).astype(int)
    return out.reset_index()


def describe_target(values) -> dict:
    """Сводка для таблицы 1 (скошенность только называется)."""
    s = pd.Series(values, dtype="float64")
    q1, median, q3 = (float(x) for x in s.quantile([0.25, 0.5, 0.75]))
    skew = float(s.skew())
    return {
        "Число наблюдений": int(s.notna().sum()),
        "Число пропусков": int(s.isna().sum()),
        "Минимум": float(s.min()),
        "Первый квартиль": q1,
        "Медиана": median,
        "Среднее": float(s.mean()),
        "Третий квартиль": q3,
        "Максимум": float(s.max()),
        "Скошенность": "вправо" if skew > 0 else "влево",
    }


LOG = logging.getLogger("eda")


def rare_groups(frame: pd.DataFrame, group: str = GROUP, share: float = RARE_SHARE) -> list:
    """Группы с долей записей меньше порога."""
    shares = frame[group].astype("object").value_counts(normalize=True)
    return list(shares[shares < share].index)


def collapse_rare(frame: pd.DataFrame, rare: list, group: str = GROUP) -> pd.DataFrame:
    """Копия таблицы, где редкие группы слиты в «прочие» (только для диаграмм)."""
    out = frame.copy()
    column = out[group].astype("object")
    out[group] = column.where(~column.isin(rare), OTHER_LABEL)
    return out


def group_table(frame: pd.DataFrame, group: str = GROUP, target: str = TARGET) -> pd.DataFrame:
    """Таблица 3: число, медиана, среднее и доля пропусков по группам."""
    grouped = frame.groupby(group, observed=True)[target]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "median": grouped.median(),
            "mean": grouped.mean(),
            "nan_share": grouped.apply(lambda s: float(s.isna().mean())),
        }
    )
    return out.sort_values("n", ascending=False).reset_index()


def pivot_median(frame: pd.DataFrame, group: str = GROUP) -> pd.DataFrame:
    """Группа x год, в ячейке медиана цели; пустая ячейка означает «нет записей»."""
    return frame.pivot_table(
        index=group,
        columns=DATE,
        values=TARGET,
        aggfunc="median",
        dropna=False,
        observed=True,
    )


def iqr_mask(values, k: float = 1.5):
    """Правило 1: вне [Q1 - k*IQR, Q3 + k*IQR]. Возвращает маску и число строк."""
    s = pd.Series(values, dtype="float64")
    q1, q3 = (float(x) for x in s.quantile([0.25, 0.75]))
    spread = q3 - q1
    mask = ((s < q1 - k * spread) | (s > q3 + k * spread)).to_numpy()
    return mask, int(mask.sum())


def z_mask(values, limit: float = 3.0):
    """Правило 2: |z| > limit. Возвращает маску и число строк."""
    s = pd.Series(values, dtype="float64")
    z = (s - s.mean()) / s.std(ddof=0)
    mask = (z.abs() > limit).to_numpy()
    return mask, int(mask.sum())


def anomaly_report(values):
    """Таблица 4 и обе маски; журнал пишет счёт каждого правила и пересечения."""
    mask_iqr, n_iqr = iqr_mask(values)
    mask_z, n_z = z_mask(values)
    both = mask_iqr & mask_z
    n_both = int(both.sum())
    total = len(mask_iqr)
    LOG.info("всего строк: %d", total)
    LOG.info("IQR отметил: %d", n_iqr)
    LOG.info("|z| > 3 отметил: %d", n_z)
    LOG.info("в обоих списках: %d", n_both)
    table = pd.DataFrame(
        {
            "Правило": ["Межквартильный размах", "|z| > 3", "Пересечение"],
            "Число отмеченных": [n_iqr, n_z, n_both],
            "Доля": [n_iqr / total, n_z / total, n_both / total],
            "Совпало со вторым": [n_both, n_both, n_both],
        }
    )
    return table, mask_iqr, mask_z


def mean_loop(values):
    total = 0.0
    count = 0
    for value in values:
        if value == value:
            total += float(value)
            count += 1
    return total / count


def mean_numpy(values):
    array = np.asarray(values, dtype="float64")
    return float(np.nanmean(array))


def median_seconds(func, values) -> float:
    """Первый прогон разогревает кэш и не входит; затем три повтора, берётся медиана."""
    func(values)
    marks = []
    for _ in range(3):
        started = time.perf_counter()
        func(values)
        marks.append(time.perf_counter() - started)
    marks.sort()
    return marks[1]


def benchmark_mean(values) -> pd.DataFrame:
    """Таблица 5. Сначала сверка результатов; при расхождении время не сравнивают."""
    array = np.asarray(values, dtype="float64")
    loop_value = mean_loop(array)
    numpy_value = mean_numpy(array)
    if not np.isclose(loop_value, numpy_value, rtol=1e-12):
        raise ValueError(f"Результаты разошлись: {loop_value} и {numpy_value}")
    loop_time = median_seconds(mean_loop, array)
    numpy_time = median_seconds(mean_numpy, array)
    return pd.DataFrame(
        {
            "Способ": ["Цикл Python", "numpy", "Отношение цикл / numpy", "Число значений"],
            "Медиана времени, с": [loop_time, numpy_time, loop_time / numpy_time, float(len(array))],
            "Значение метрики": [loop_value, numpy_value, np.nan, np.nan],
        }
    )
