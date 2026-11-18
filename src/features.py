"""Витрина признаков лабораторной работы 3: текст, категории, динамика, агрегаты, отбор.

Запуск из корня проекта: python -m src.features
Читает data/processed/clean.parquet, пишет data/processed/features.parquet и FEATURES.md.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "clean.parquet"
OUT = ROOT / "data" / "processed" / "features.parquet"
DOC = ROOT / "FEATURES.md"
DICTIONARY = ROOT / "docs" / "columns_dictionary.csv"

KEY = "id_uc_g0"
TARGET_SRC = "gh_pop_tot_2025"
TARGET = "pop_tot"
TARGET_CLASS = "pop_above_median"
COUNTRY_SRC = "gc_cnt_gad_2025"
SEED = 42
TEST_SIZE = 0.2
CONST_SHARE = 0.95
CORR_LIMIT = 0.9

RAW_NUMERIC = [
    "ex_010_bus_2025",
    "ex_010_nre_2025",
    "gc_uca_km2_2025",
    "ge_bio_a24_2025",
    "ge_elv_avg_2025",
    "gh_age_s05_2025",
    "gh_age_s15_2025",
    "gr_avg_grn_2025",
    "gr_shb_grn_2025",
    "xst_a30_2025",
]
DYNAMICS = [
    ("pm25", "em_pm2_con", 2000, 2020),
    ("temp", "cl_b01_cur", 1970, 2010),
    ("precip", "cl_b12_cur", 1970, 2010),
]
KEYWORDS = {
    "tpa_park": r"park|parque",
    "tpa_reserve": r"reserve|reserva",
    "tpa_protection": r"protect|protecc",
    "tpa_municipal": r"municipal",
    "tpa_forest": r"forest|bosque|floresta",
}
TOKEN = re.compile(r"[^\W\d_]{3,}")
STOP = {
    "the", "and", "for", "del", "las", "los", "dos", "das", "con", "que", "por", "para",
    "una", "uno", "with", "from",
}
DESCRIPTIONS = {
    "country": "страна города (gc_cnt_gad_2025), пропуски заменены на unknown",
    "income_group": "группа дохода страны (gc_dev_wig_2025)",
    "soil_main": "основной класс почв (ge_scl_mpc_2023)",
    "gwr_gwb": "категория подземных вод (wa_gwr_gwb_2025)",
    "gwr_mma": "класс из wa_gwr_mma_2025 (значения вида Medium - Very Low)",
    "eco_freq": "доля городов с той же экорегиональной категорией (ge_eco_cla_2025)",
    "soil_list_freq": "доля городов с тем же списком почв (ge_scl_lst_2023)",
    "is_capital": "столица страны, флаг 0/1 (gc_ucm_cap)",
    "pm25_2020": "PM2.5, мкг/м3, средневзвешенная по населению, 2020 (em_pm2_con_2020)",
    "pm25_delta": "изменение PM2.5 за 2000-2020 (em_pm2_con_2020 минус em_pm2_con_2000)",
    "temp_2010": "среднегодовая температура десятилетия, °C, 2010 (cl_b01_cur_2010)",
    "temp_delta": "изменение температуры за 1970-2010 (cl_b01_cur_2010 минус cl_b01_cur_1970)",
    "precip_2010": "среднегодовые осадки, мм/год, 2010 (cl_b12_cur_2010)",
    "precip_delta": "изменение осадков за 1970-2010 (cl_b12_cur_2010 минус cl_b12_cur_1970)",
    "country_n": "число городов страны в наборе",
    "area_country_median": "медиана площади (gc_uca_km2_2025) по стране",
    "area_dev": "площадь города минус медиана площади по стране",
    "pm25_country_median": "медиана PM2.5 за 2020 по стране",
    "pm25_dev": "PM2.5 города за 2020 минус медиана по стране",
    "tpa_n": "число наземных охраняемых территорий: элементов через « ; » в ns_tpa_nam_2025, 0 если пусто",
    "mpa_n": "число морских охраняемых территорий: элементов через « ; » в ns_mpa_nam_2025, 0 если пусто",
    "alt_names_n": "число названий города через «;» в gc_ucn_lis_2025",
    "name_len": "длина основного названия, символов (gc_ucn_mai_2025)",
    "tpa_des_len": "длина описаний наземных охраняемых территорий, символов (ns_tpa_des_2025)",
    "tpa_des_words": "число слов в описаниях после очистки (не короче 3 букв, без стоп-слов)",
    "has_tpa": "есть наземная охраняемая территория (ns_tpa_nam_2025 не пусто)",
    "has_mpa": "есть морская охраняемая территория (ns_mpa_nam_2025 не пусто)",
    "tpa_park": "в описаниях есть park или parque",
    "tpa_reserve": "в описаниях есть reserve или reserva",
    "tpa_protection": "в описаниях есть protect или protecc",
    "tpa_municipal": "в описаниях есть municipal",
    "tpa_forest": "в описаниях есть forest, bosque или floresta",
    "tpa_kw_n": "сколько из пяти шаблонов (park, reserve, protect, municipal, forest) сработало",
}
LOG = logging.getLogger("features")


def split_mask(keys, test_size: float = TEST_SIZE, seed: int = SEED) -> np.ndarray:
    """Маска тестовой части: фиксированная перестановка ключей, не зависит от порядка строк."""
    keys = np.asarray(keys)
    order = np.argsort(keys, kind="stable")
    shuffled = np.random.default_rng(seed).permutation(len(keys))
    n_test = int(round(len(keys) * test_size))
    mask = np.zeros(len(keys), dtype=bool)
    mask[order[shuffled[:n_test]]] = True
    return mask


def clean_text(text) -> str:
    """Нижний регистр, ё заменяется на е."""
    return str(text).lower().replace("ё", "е")


def tokens(text) -> list:
    """Слова не короче трёх букв без стоп-слов."""
    return [w for w in TOKEN.findall(clean_text(text)) if w not in STOP]


def count_items(series: pd.Series, sep: str = ";") -> pd.Series:
    """Число непустых элементов в строке-списке; пустое значение даёт 0."""
    pattern = r"\s*" + re.escape(sep) + r"\s*"
    return series.fillna("").astype(str).map(
        lambda s: sum(1 for part in re.split(pattern, s) if part.strip())
    ).astype("int32")


def _cat(series: pd.Series) -> pd.Series:
    return series.astype("object").fillna("unknown").astype("category")


def _freq(series: pd.Series) -> pd.Series:
    values = series.astype("object").fillna("unknown")
    return values.map(values.value_counts(normalize=True)).astype("float32")


def category_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["country"] = _cat(df[COUNTRY_SRC])
    out["income_group"] = _cat(df["gc_dev_wig_2025"])
    out["soil_main"] = _cat(df["ge_scl_mpc_2023"])
    out["gwr_gwb"] = _cat(df["wa_gwr_gwb_2025"])
    out["gwr_mma"] = _cat(df["wa_gwr_mma_2025"])
    out["eco_freq"] = _freq(df["ge_eco_cla_2025"])
    out["soil_list_freq"] = _freq(df["ge_scl_lst_2023"])
    out["is_capital"] = df["gc_ucm_cap"].astype("int8")
    return out


def dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Замена даты: значение за последний год и изменение за период."""
    out = pd.DataFrame(index=df.index)
    for short, stem, first, last in DYNAMICS:
        before = df[f"{stem}_{first}"].astype("float64")
        after = df[f"{stem}_{last}"].astype("float64")
        out[f"{short}_{last}"] = after
        out[f"{short}_delta"] = after - before
    return out


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты по стране для других столбцов; цель в агрегаты не входит."""
    country = df[COUNTRY_SRC].astype("object").fillna("unknown")
    out = pd.DataFrame(index=df.index)
    out["country_n"] = country.map(country.value_counts()).astype("int32")
    area = df["gc_uca_km2_2025"].astype("float64")
    pm25 = df["em_pm2_con_2020"].astype("float64")
    out["area_country_median"] = area.groupby(country).transform("median")
    out["area_dev"] = area - out["area_country_median"]
    out["pm25_country_median"] = pm25.groupby(country).transform("median")
    out["pm25_dev"] = pm25 - out["pm25_country_median"]
    return out


def text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    tpa_names = df["ns_tpa_nam_2025"]
    mpa_names = df["ns_mpa_nam_2025"]
    des = df["ns_tpa_des_2025"].fillna("").astype(str).str.lower()
    out["tpa_n"] = count_items(tpa_names)
    out["mpa_n"] = count_items(mpa_names)
    out["alt_names_n"] = count_items(df["gc_ucn_lis_2025"])
    out["name_len"] = df["gc_ucn_mai_2025"].fillna("").astype(str).str.len().astype("int32")
    out["tpa_des_len"] = des.str.len().astype("int32")
    out["tpa_des_words"] = des.map(lambda text: len(tokens(text))).astype("int32")
    out["has_tpa"] = tpa_names.notna()
    out["has_mpa"] = mpa_names.notna()
    hits = pd.Series(0, index=df.index, dtype="int8")
    for name, pattern in KEYWORDS.items():
        found = des.str.contains(pattern, regex=True)
        out[name] = found
        hits = hits + found.astype("int8")
    out["tpa_kw_n"] = hits
    return out


def raw_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df[RAW_NUMERIC].astype("float32")


def build_features(df: pd.DataFrame) -> dict:
    """Части витрины по источникам, в порядке, который важен при отборе пар."""
    return {
        "категории": category_features(df),
        "дата (динамика)": dynamics_features(df),
        "исходные числовые": raw_numeric(df),
        "агрегаты по группе": aggregate_features(df),
        "текст": text_features(df),
    }


def select_features(frame: pd.DataFrame, test_mask) -> tuple:
    """Отбор только по обучающей части и без цели: почти константные и повторяющие друг друга."""
    train = frame.loc[~np.asarray(test_mask)]
    dropped = {}
    for column in frame.columns:
        top = train[column].value_counts(normalize=True, dropna=False).iloc[0]
        if top > CONST_SHARE:
            dropped[column] = f"одно значение занимает {top:.1%} строк"
    candidates = [c for c in frame.columns if c not in dropped]
    numeric = [
        c for c in candidates
        if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_bool_dtype(train[c])
    ]
    corr = train[numeric].astype("float64").corr(method="pearson").abs()
    kept_numeric = []
    for column in numeric:
        partner = next((k for k in kept_numeric if corr.loc[column, k] > CORR_LIMIT), None)
        if partner is None:
            kept_numeric.append(column)
        else:
            dropped[column] = f"|r| = {corr.loc[column, partner]:.2f} с {partner}"
    kept = [c for c in candidates if c not in numeric or c in kept_numeric]
    return kept, dropped


def word_bigram_counts(texts: pd.Series) -> tuple:
    """Частотный словарь слов и биграмм по непустым текстам."""
    words, pairs = Counter(), Counter()
    for text in texts.dropna():
        found = tokens(text)
        words.update(found)
        pairs.update(zip(found, found[1:]))
    return words, pairs


def word_share_by_group(texts: pd.Series, labels: pd.Series, words: list) -> pd.DataFrame:
    """Доля записей группы, в тексте которых встречается слово (замена долей по датам)."""
    token_sets = texts.map(lambda text: set(tokens(text)))
    rows = {}
    for group in labels.unique():
        inside = token_sets[(labels == group).to_numpy()]
        rows[group] = {w: float(inside.map(lambda found: w in found).mean()) for w in words}
    return pd.DataFrame(rows).T


def tfidf_top_words(texts: pd.Series, labels: pd.Series, groups: list, top: int = 5) -> dict:
    """Слова, у которых средний вес TF-IDF в группе выше среднего по набору."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    vectorizer = TfidfVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None)
    matrix = vectorizer.fit_transform(texts)
    names = np.array(vectorizer.get_feature_names_out())
    overall = np.asarray(matrix.mean(axis=0)).ravel()
    result = {}
    for group in groups:
        rows = np.flatnonzero((labels == group).to_numpy())
        inside = np.asarray(matrix[rows].mean(axis=0)).ravel()
        best = np.argsort(inside - overall)[::-1][:top]
        result[group] = [(str(names[i]), float(inside[i]), float(overall[i])) for i in best]
    return result


def write_features_md(path, frame, kept, dropped, source_of, median) -> None:
    """FEATURES.md: ключ и цели, колонки витрины, отброшенные при отборе."""
    raw_info = {}
    if DICTIONARY.exists():
        dic = pd.read_csv(DICTIONARY).set_index("column")
        for column in RAW_NUMERIC:
            if column in dic.index:
                raw_info[column] = f"{dic.loc[column, 'indicator']}, {dic.loc[column, 'unit']}"
    median_text = f"{median:,.0f}".replace(",", " ")
    lines = [
        "# FEATURES: витрина признаков (data/processed/features.parquet)",
        "",
        f"Одна строка витрины соответствует одному городу ({len(frame)} строк). Колонки после сдачи не переименовываются.",
        "",
        "## Ключ и цели",
        "",
        "| Колонка | Тип | Как получена |",
        "|---|---|---|",
        f"| {KEY} | int | ключ города из clean.parquet |",
        f"| {TARGET} | float64 | численность населения города, 2025 ({TARGET_SRC}), цель регрессии |",
        f"| {TARGET_CLASS} | int8 | 1, если {TARGET} выше медианы обучающей части ({median_text}), цель классификации |",
        "",
        "## Колонки витрины",
        "",
        "| Колонка | Тип | Источник | Как получена | Модель |",
        "|---|---|---|---|---|",
    ]
    for column in kept:
        how = DESCRIPTIONS.get(column)
        if how is None:
            how = f"исходный столбец {column}" + (f": {raw_info[column]}" if column in raw_info else "")
        lines.append(f"| {column} | {frame[column].dtype} | {source_of[column]} | {how} | да |")
    lines += ["", "## Отброшено при отборе", "", "| Колонка | Источник | Причина |", "|---|---|---|"]
    for column, reason in dropped.items():
        lines.append(f"| {column} | {source_of[column]} | {reason} |")
    lines += [
        "",
        "## Служебное",
        "",
        f"- Разбиение: функция split_mask в src/features.py, случайные {TEST_SIZE:.0%} ключей в тест, seed {SEED}.",
        f"- Отбор (доля одного значения больше {CONST_SHARE:.0%}; |r| больше {CORR_LIMIT} внутри пары, остаётся первая по порядку колонка) считается только по обучающей части и без цели.",
        "- Агрегаты по стране считаются по признакам всех строк, но без цели.",
        "- Колонки с населением и производные от него (gh_pop_*, gc_pop_*, ex_*_pop, em_co2_pec) в признаки не входят.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = pd.read_parquet(CLEAN)
    test = split_mask(df[KEY].to_numpy())
    parts = build_features(df)
    frame = pd.concat(list(parts.values()), axis=1)
    source_of = {column: name for name, part in parts.items() for column in part.columns}
    kept, dropped = select_features(frame, test)
    target = df[TARGET_SRC].astype("float64")
    median = float(target[~test].median())
    LOG.info("строк: %d, тест: %d", len(df), int(test.sum()))
    LOG.info("признаков до отбора: %d, после: %d", frame.shape[1], len(kept))
    for column, reason in dropped.items():
        LOG.info("отброшено %s: %s", column, reason)
    features = frame[kept].copy()
    for column in features.columns:
        if features[column].dtype == "float64":
            features[column] = features[column].astype("float32")
    out = pd.concat(
        [
            df[[KEY]].reset_index(drop=True),
            pd.DataFrame(
                {TARGET: target.to_numpy(), TARGET_CLASS: (target > median).astype("int8").to_numpy()}
            ),
            features.reset_index(drop=True),
        ],
        axis=1,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    write_features_md(DOC, features, kept, dropped, source_of, median)
    LOG.info("записано: %s %s", OUT, out.shape)


if __name__ == "__main__":
    main()
