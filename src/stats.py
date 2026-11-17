"""Статистические расчёты лабораторной работы 3: бутстрап, тесты, корреляции."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

ALPHA = 0.05
N_BOOT = 2000
SEED = 42


def _clean(values) -> np.ndarray:
    """Массив float без пропусков."""
    array = np.asarray(values, dtype="float64")
    return array[~np.isnan(array)]


def bootstrap_median(values, n_boot: int = N_BOOT, seed: int = SEED):
    """95-процентный интервал медианы: перцентили 2,5 и 97,5 бутстрап-медиан."""
    array = _clean(values)
    rng = np.random.default_rng(seed)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        medians[i] = np.median(rng.choice(array, size=len(array)))
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_diff_median(a, b, n_boot: int = N_BOOT, seed: int = SEED):
    """Разность медиан a - b и её 95-процентный бутстрап-интервал."""
    a = _clean(a)
    b = _clean(b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = np.median(rng.choice(a, size=len(a))) - np.median(rng.choice(b, size=len(b)))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(np.median(a) - np.median(b)), float(low), float(high)


def distribution_summary(values, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Строки таблицы 2: среднее, медиана, асимметрия до и после log1p, бутстрап."""
    array = _clean(values)
    series = pd.Series(array)
    low, high = bootstrap_median(array, n_boot, seed)
    return {
        "n": int(len(array)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "skew_raw": float(series.skew()),
        "skew_log1p": float(np.log1p(series).skew()),
        "ci_low": low,
        "ci_high": high,
        "n_boot": int(n_boot),
    }


def mannwhitney(a, b) -> dict:
    """Тест Манна-Уитни, двусторонний."""
    a = _clean(a)
    b = _clean(b)
    result = sp.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "u": float(result.statistic),
        "p": float(result.pvalue),
    }


def holm(p_values):
    """Поправка Холма: маска значимых и скорректированные p-значения."""
    p = np.asarray(p_values, dtype="float64")
    reject, adjusted, _, _ = multipletests(p, alpha=ALPHA, method="holm")
    return reject, adjusted


def one_vs_rest(frame: pd.DataFrame, group_col: str, target: str) -> pd.DataFrame:
    """Каждая группа против остальных (Манн-Уитни) с поправкой Холма по всем группам."""
    labels = frame[group_col].astype("object")
    rows = []
    for name in labels.unique():
        mask = (labels == name).to_numpy()
        res = mannwhitney(frame.loc[mask, target], frame.loc[~mask, target])
        rows.append(
            {
                "group": name,
                "n": res["n_a"],
                "median": res["median_a"],
                "median_rest": res["median_b"],
                "diff": res["median_a"] - res["median_b"],
                "p_raw": res["p"],
            }
        )
    out = pd.DataFrame(rows)
    reject, adjusted = holm(out["p_raw"])
    out["p_holm"] = adjusted
    out["significant"] = reject
    return out.sort_values("p_raw").reset_index(drop=True)


def kruskal_groups(groups: dict) -> dict:
    """Краскел-Уоллис по словарю {имя: значения}; ε² = (H - k + 1) / (n - k)."""
    arrays = [_clean(v) for v in groups.values()]
    h, p = sp.kruskal(*arrays)
    k = len(arrays)
    n = sum(len(a) for a in arrays)
    return {"h": float(h), "p": float(p), "k": k, "n": int(n), "eps2": float((h - k + 1) / (n - k))}


def pairwise_holm(groups: dict) -> pd.DataFrame:
    """Попарные Манн-Уитни по всем парам групп с поправкой Холма."""
    names = list(groups)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res = mannwhitney(groups[a], groups[b])
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "n_a": res["n_a"],
                    "n_b": res["n_b"],
                    "median_a": res["median_a"],
                    "median_b": res["median_b"],
                    "diff": res["median_a"] - res["median_b"],
                    "p_raw": res["p"],
                }
            )
    out = pd.DataFrame(rows)
    reject, adjusted = holm(out["p_raw"])
    out["p_holm"] = adjusted
    out["significant"] = reject
    return out.sort_values("p_raw").reset_index(drop=True)


def paired_share_test(before, after, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Доля со свойством до и после: z-тест двух долей, Мак-Немар и парный бутстрап.

    z-тест считает разность «до - после», поэтому знак z обратен знаку diff_pp.
    """
    before = np.asarray(before, dtype=bool)
    after = np.asarray(after, dtype=bool)
    n = len(before)
    z, p_z = proportions_ztest(np.array([before.sum(), after.sum()]), np.array([n, n]))
    only_before = int((before & ~after).sum())
    only_after = int((~before & after).sum())
    table = [
        [int((before & after).sum()), only_before],
        [only_after, int((~before & ~after).sum())],
    ]
    exact = (only_before + only_after) < 25
    mc = mcnemar(table, exact=exact, correction=True)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        index = rng.integers(0, n, size=n)
        diffs[i] = after[index].mean() - before[index].mean()
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "n": int(n),
        "share_before": float(before.mean()),
        "share_after": float(after.mean()),
        "diff_pp": float(100 * (after.mean() - before.mean())),
        "ci_low_pp": float(100 * low),
        "ci_high_pp": float(100 * high),
        "z": float(z),
        "p_z": float(p_z),
        "only_before": only_before,
        "only_after": only_after,
        "mcnemar_exact": bool(exact),
        "p_mcnemar": float(mc.pvalue),
    }


def corr_gap(frame: pd.DataFrame, columns: list):
    """Матрицы Пирсона и Спирмена и их разность (Пирсон минус Спирмен)."""
    sub = frame[columns]
    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    return pearson, spearman, pearson - spearman


def largest_gap(pearson: pd.DataFrame, spearman: pd.DataFrame) -> dict:
    """Пара с наибольшим по модулю расхождением выше диагонали."""
    gap = pearson - spearman
    upper = gap.where(np.triu(np.ones(gap.shape, dtype=bool), k=1))
    stacked = upper.stack()
    first, second = stacked.abs().idxmax()
    return {
        "a": first,
        "b": second,
        "pearson": float(pearson.loc[first, second]),
        "spearman": float(spearman.loc[first, second]),
        "gap": float(stacked.loc[(first, second)]),
    }


def simpson_table(frame: pd.DataFrame, flag: str, target: str, group: str, min_each: int = 3):
    """Разность медиан цели (флаг 1 минус флаг 0) во всём наборе и в каждой группе."""
    data = frame[[flag, target]].assign(**{group: frame[group].astype("object")})
    levels = sorted(data[flag].dropna().unique())
    low, high = levels[0], levels[-1]
    overall = data.groupby(flag)[target].median()
    overall_diff = float(overall[high] - overall[low])
    grouped = data.groupby([group, flag])[target].agg(["median", "size"]).unstack(flag)
    rows = []
    for name, row in grouped.iterrows():
        n_low = row[("size", low)]
        n_high = row[("size", high)]
        if not (n_low >= min_each and n_high >= min_each):
            continue
        rows.append(
            {
                group: name,
                "n_flag0": int(n_low),
                "n_flag1": int(n_high),
                "diff": float(row[("median", high)] - row[("median", low)]),
            }
        )
    return overall_diff, pd.DataFrame(rows).sort_values("diff").reset_index(drop=True)
