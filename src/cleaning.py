"""Функции очистки данных GHS-UCDB."""
import logging
import re
import unicodedata

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BAD = re.compile(r"Ã[\u0080-\u00BF]|[\u0080-\u009F]")
_THOUSANDS = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")
_YEAR = re.compile(r"_(?:19|20)\d{2}")

PLACEHOLDERS = {"", "-", "--", "n/a", "na", "nan", "none", "null", "unknown", "no data"}
COLUMN_RENAMES = {"NS_TPA_DES_202": "NS_TPA_DES_2025"}
MAX_MISSING_SHARE = 0.99
CENSORED = {">5": 5.0, "<-5": -5.0}
OUTLIER_COLUMNS = ["GC_POP_TOT_2025", "GC_UCA_KM2_2025"]


def fix_mojibake(s):
    """Чинит строки с двойным кодированием UTF-8; корректные строки не меняет."""
    if not isinstance(s, str) or s.isascii():
        return s
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def _is_text(s):
    return s.dtype == "object" or str(s.dtype) in ("str", "string")


def fix_text_columns(df):
    """Применяет fix_mojibake ко всем текстовым столбцам, возвращает копию."""
    out = df.copy()
    for c in out.columns:
        if _is_text(out[c]):
            out[c] = out[c].map(fix_mojibake)
    return out


def _looks_broken(s):
    return isinstance(s, str) and _BAD.search(s) is not None


def count_mojibake(df):
    """Число значений по столбцам, всё ещё похожих на испорченную кодировку."""
    res = {}
    for c in df.columns:
        if _is_text(df[c]):
            n = int(df[c].map(_looks_broken).sum())
            if n:
                res[c] = n
    return res


def rename_truncated_columns(df):
    """Исправляет обрезанные имена столбцов по словарю COLUMN_RENAMES."""
    return df.rename(columns=COLUMN_RENAMES)


def drop_service_columns(df):
    """Удаляет служебные столбцы вида 'Unnamed: N' (следы выгрузки индекса)."""
    keep = [c for c in df.columns if not str(c).startswith("Unnamed")]
    return df[keep]


def drop_key_duplicates(df, key, flag_col="dup_key_flag"):
    """Оставляет первую запись по ключу, ключи с дублями помечает флагом."""
    out = df.copy()
    out[flag_col] = out[key].duplicated(keep=False)
    return out.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def _to_number(v):
    """Строку с числом (в т.ч. '1,275') превращает в float, заглушки в NaN."""
    if not isinstance(v, str):
        return v
    t = v.strip()
    if t.lower() in PLACEHOLDERS:
        return float("nan")
    if _THOUSANDS.match(t):
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return v


def find_numeric_text_columns(df, min_share=0.95):
    """Текстовые столбцы, значения которых почти все являются числами."""
    res = []
    for c in df.columns:
        if not _is_text(df[c]):
            continue
        s = df[c].dropna().map(_to_number)
        is_num = s.map(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
        if len(s) and is_num.mean() >= min_share:
            res.append(c)
    return res


def fix_numeric_text(df, columns):
    """Приводит указанные столбцы к числам; при неразобранном значении падает."""
    out = df.copy()
    for c in columns:
        out[c] = pd.to_numeric(out[c].map(_to_number), errors="raise")
    return out


def fix_censored_numeric(df, columns, mapping=CENSORED, flag_suffix="_cens"):
    """Числа, записанные текстом с пределами ('>5'), приводит к float и добавляет флаг."""
    out = df.copy()
    for c in columns:
        raw = out[c]
        out[c + flag_suffix] = raw.map(
            lambda v: isinstance(v, str) and v.strip() in mapping
        ).astype(bool)
        out[c] = pd.to_numeric(
            raw.map(lambda v: mapping.get(v.strip(), v) if isinstance(v, str) else v),
            errors="raise",
        )
    return out


def find_censored_columns(df, mapping=CENSORED):
    """Текстовые столбцы, где все нечисловые значения - пределы вида '>5'."""
    res = []
    for c in df.columns:
        if not _is_text(df[c]):
            continue
        s = df[c].dropna().astype("string").str.strip()
        is_cens = s.isin(list(mapping))
        rest = pd.to_numeric(s[~is_cens], errors="coerce")
        if is_cens.any() and rest.notna().all():
            res.append(c)
    return res


def find_sparse_columns(df, key, max_share=MAX_MISSING_SHARE):
    """Имена столбцов, где доля пропусков больше max_share (ключ не трогаем)."""
    share = df.isna().mean()
    return [c for c in df.columns if c != key and share[c] > max_share]


def clean_sheet(df, key):
    """Очищает один тематический лист; возвращает (таблица, статистика)."""
    st = {"строк_было": len(df), "столбцов_было": df.shape[1]}

    out = drop_service_columns(rename_truncated_columns(df))
    st["служебных_удалено"] = df.shape[1] - out.shape[1]

    broken = sum(count_mojibake(out).values())
    out = fix_text_columns(out)
    st["кодировка_исправлено"] = broken - sum(count_mojibake(out).values())

    cens = find_censored_columns(out)
    out = fix_censored_numeric(out, cens)
    st["обрезанных_пределов"] = int(sum(out[c + "_cens"].sum() for c in cens))
    num = find_numeric_text_columns(out)
    out = fix_numeric_text(out, num)
    st["текст_в_число"] = len(cens) + len(num)

    n = len(out)
    if out[key].duplicated().any():
        out = drop_key_duplicates(out, key)
    st["дублей_удалено"] = n - len(out)

    sparse = find_sparse_columns(out, key)
    out = out.drop(columns=sparse)
    st["пустых_удалено"] = len(sparse)
    st["пустые_столбцы"] = sparse
    st["строк_стало"], st["столбцов_стало"] = len(out), out.shape[1]
    logger.info("лист очищен: строк %d->%d, столбцов %d->%d",
                st["строк_было"], st["строк_стало"], st["столбцов_было"], st["столбцов_стало"])
    return out, st


def merge_sheets(sheets, key, base="GENERAL_CHARACTERISTICS"):
    """Левое соединение листов с базовым по ключу; проверяет число строк."""
    result = sheets[base]
    report = []
    for name in sorted(sheets):
        d = sheets[name]
        if name == base:
            continue
        a = result.set_index(key)
        common = [c for c in d.columns if c in a.columns and c != key]
        b = d.set_index(key).reindex(a.index)
        mism = {c: int((~(a[c].eq(b[c]).fillna(False).astype(bool)
                          | (a[c].isna() & b[c].isna()))).sum()) for c in common}
        n0 = len(result)
        result = result.merge(d.drop(columns=common), on=key, how="left", validate="one_to_one")
        report.append({
            "лист": name, "строк_до": n0, "строк_после": len(result),
            "нет_в_базе": int((~d[key].isin(a.index)).sum()),
            "нет_в_листе": int((~a.index.isin(d[key])).sum()),
            "повторных_столбцов": len(common),
            "расхождений_в_повторах": sum(mism.values()),
        })
        if len(result) != n0:
            raise ValueError(f"{name}: число строк изменилось {n0} -> {len(result)}")
    return result, pd.DataFrame(report)


def attr_id(col):
    """'EX_L10_SHP_2030' -> 'EX_L10_SHP_XXXX' (как в листе Index)."""
    return _YEAR.sub("_XXXX", str(col))


def column_units(columns, index_df):
    """Единицы измерения столбцов по листу Index (None, если описания нет)."""
    d = index_df.dropna(subset=["Attribute ID"]).copy()
    d["Attribute ID"] = d["Attribute ID"].astype("string").str.strip()
    unit = d.drop_duplicates("Attribute ID").set_index("Attribute ID")["Unit"]
    res = {}
    for c in columns:
        v = unit.get(attr_id(c))
        res[c] = None if v is None or pd.isna(v) else str(v)
    return res


def find_constant_columns(df, key):
    """Столбцы, где не более одного уникального значения (ключ не трогаем)."""
    return [c for c in df.columns if c != key and df[c].nunique(dropna=True) <= 1]


def drop_constant_columns(df, key):
    """Удаляет столбцы с одним значением; возвращает (таблица, список удалённых)."""
    const = find_constant_columns(df, key)
    return df.drop(columns=const), const


def normalize_column_names(df):
    """Нижний регистр, всё кроме букв и цифр заменяется на '_'."""
    new = (df.columns.astype(str).str.strip().str.lower()
           .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))
    if new.duplicated().any():
        raise ValueError("после нормализации имена столбцов совпали")
    out = df.copy()
    out.columns = new
    return out


def outlier_masks(s, k=1.5, z=3.0):
    """Маски выбросов тремя способами: IQR, 3 сигмы, IQR на log1p."""
    x = s.astype("float64")
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    m_iqr = (x < q1 - k * iqr) | (x > q3 + k * iqr)
    m_z = ((x - x.mean()) / x.std()).abs() > z
    lx = np.log1p(x.clip(lower=0))
    l1, l3 = lx.quantile([0.25, 0.75])
    m_log = (lx < l1 - k * (l3 - l1)) | (lx > l3 + k * (l3 - l1))
    return {"iqr": m_iqr, "3sigma": m_z, "iqr_log": m_log}


def add_outlier_flags(df, columns, flag_suffix="_out"):
    """Добавляет флаги выбросов (IQR на log1p); значения не меняются."""
    out = df.copy()
    for c in columns:
        out[c + flag_suffix] = outlier_masks(out[c])["iqr_log"].fillna(False).astype(bool)
    return out


def flag_out_of_range(df, columns, lo=None, hi=None, flag_col="out_of_range"):
    """Один флаг на строку: хотя бы в одном из столбцов значение вне [lo, hi]."""
    out = df.copy()
    bad = pd.Series(False, index=out.index)
    for c in columns:
        if hi is not None:
            bad |= (out[c] > hi).fillna(False)
        if lo is not None:
            bad |= (out[c] < lo).fillna(False)
    out[flag_col] = bad
    return out


def build_clean(sheets_clean, key, index_df):
    """Соединяет очищенные листы и достраивает итоговую таблицу.

    Возвращает (таблица с нормализованными именами столбцов, отчёт-словарь).
    """
    df, join_report = merge_sheets(sheets_clean, key)
    cols_joined = df.shape[1]

    df, const = drop_constant_columns(df, key)

    units = column_units(df.columns, index_df)
    pct = [c for c in df.select_dtypes("number").columns if units.get(c) == "%"]
    df = flag_out_of_range(df, pct, lo=0, hi=100)
    n_range = int(df["out_of_range"].sum())
    df = add_outlier_flags(df, OUTLIER_COLUMNS)
    out_counts = {c: int(df[c + "_out"].sum()) for c in OUTLIER_COLUMNS}

    df = normalize_column_names(df)
    report = {
        "join": join_report,
        "столбцов_после_соединения": cols_joined,
        "константных_удалено": const,
        "столбцов_в_процентах": len(pct),
        "строк_с_долей_вне_0_100": n_range,
        "выбросы_log_iqr": out_counts,
        "key": key.lower(),
        "shape": df.shape,
    }
    logger.info("итоговая таблица: %s, константных удалено %d, долей вне 0-100: %d строк",
                df.shape, len(const), n_range)
    return df, report


INT_LIMITS = {"int16": (-32768, 32767), "int32": (-(2**31), 2**31 - 1)}


def _norm_unit(u):
    """Приводит единицу к нормальной форме (NFKC), чтобы 'µ' совпадали."""
    return None if u is None else unicodedata.normalize("NFKC", str(u)).strip()


def apply_schema(df, units, float32_units, int16_units, int32_units, cat_max_share=0.5):
    """Назначает типы по единицам измерения; возвращает (таблица, план {столбец: тип})."""
    f32 = {_norm_unit(u) for u in float32_units}
    i16 = {_norm_unit(u) for u in int16_units}
    i32 = {_norm_unit(u) for u in int32_units}
    out = df.copy()
    plan = {}
    for c in out.columns:
        s = out[c]
        u = _norm_unit(units.get(c))
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_float_dtype(s) and u in f32:
            plan[c] = "float32"
        elif pd.api.types.is_integer_dtype(s) and u in i16:
            plan[c] = "int16"
        elif pd.api.types.is_integer_dtype(s) and u in i32:
            plan[c] = "int32"
        elif _is_text(s):
            nn = s.dropna()
            if len(nn) and nn.nunique() / len(nn) < cat_max_share:
                plan[c] = "category"
    for c, t in plan.items():
        if t in INT_LIMITS:
            lo, hi = INT_LIMITS[t]
            if out[c].min() < lo or out[c].max() > hi:
                raise ValueError(f"{c}: значения не помещаются в {t}")
        out[c] = out[c].astype(t)
    logger.info("схема типов применена к %d столбцам", len(plan))
    return out, plan
