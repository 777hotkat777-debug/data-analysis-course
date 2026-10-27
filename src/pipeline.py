"""Сквозной сценарий обработки GHS-UCDB.

Данные идут только в одну сторону: raw -> interim -> processed.
  serialize  data/raw/*.xlsx          -> data/interim/sheets_raw/*.parquet
  clean      data/interim/sheets_raw  -> data/interim/clean_untyped.parquet
  typed      clean_untyped.parquet    -> data/processed/clean.parquet

Запуск из корня проекта:
    python -m src.pipeline                    # все этапы
    python -m src.pipeline --stage clean      # один этап
    python -m src.pipeline --sample 200       # отладочный срез (отдельные файлы)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src import config
from src.cleaning import apply_schema, build_clean, clean_sheet, column_units

log = logging.getLogger("pipeline")
STAGES = ("serialize", "clean", "typed")


def setup_logging(log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def paths(sample):
    if sample:
        d = config.INTERIM_DIR / "sample"
        return {"untyped": d / "clean_untyped.parquet", "final": d / "clean.parquet"}
    return {"untyped": config.CLEAN_UNTYPED, "final": config.CLEAN_PARQUET}


def check_sheets(sheets):
    for need in (config.BASE_SHEET, "Index"):
        if need not in sheets:
            raise ValueError(f"в данных нет обязательного листа {need}")
    if config.KEY not in sheets[config.BASE_SHEET].columns:
        raise ValueError(f"в листе {config.BASE_SHEET} нет ключа {config.KEY}")


def read_book(path):
    if not path.exists():
        raise FileNotFoundError(f"исходный файл не найден: {path}")
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"ожидался файл .xlsx, получен {path.suffix}")
    log.info("чтение %s", path.name)
    t0 = time.perf_counter()
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        raise ValueError(f"не удалось прочитать книгу (файл повреждён?): {e}") from e
    log.info("прочитано листов: %d за %.1f с", len(sheets), time.perf_counter() - t0)
    check_sheets(sheets)
    return sheets


def load_sheets(sheets_dir):
    files = sorted(sheets_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"нет файлов в {sheets_dir}: сначала выполните этап serialize")
    sheets = {p.stem: pd.read_parquet(p) for p in files}
    check_sheets(sheets)
    return sheets


def stage_serialize(src_path, sheets_dir):
    sheets = read_book(src_path)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for old in sheets_dir.glob("*.parquet"):
        old.unlink()
    for name, d in sheets.items():
        if name in config.SERVICE_SHEETS:
            d = d.astype("string")
        d.to_parquet(sheets_dir / f"{name}.parquet")
    log.info("serialize: %d листов -> %s", len(sheets), sheets_dir)


def stage_clean(sheets_dir, out_path, sample):
    sheets = load_sheets(sheets_dir)
    cleaned = {}
    for name, d in sheets.items():
        if name in config.SERVICE_SHEETS:
            continue
        if sample:
            d = d.head(sample)
        cleaned[name], st = clean_sheet(d, config.KEY)
        log.info("%-24s строк %d->%d | столбцов %d->%d | дублей %d | пустых столбцов %d",
                 name, st["строк_было"], st["строк_стало"], st["столбцов_было"],
                 st["столбцов_стало"], st["дублей_удалено"], st["пустых_удалено"])
    df, _ = build_clean(cleaned, config.KEY, sheets["Index"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd")
    log.info("clean: %s, %s", out_path.name, df.shape)


def stage_typed(sheets_dir, in_path, out_path):
    if not in_path.exists():
        raise FileNotFoundError(f"нет файла {in_path}: сначала выполните этап clean")
    df = pd.read_parquet(in_path)
    index_df = pd.read_parquet(sheets_dir / "Index.parquet")
    units = column_units([c.upper() for c in df.columns], index_df)
    units = {k.lower(): v for k, v in units.items()}
    typed, plan = apply_schema(df, units, config.FLOAT32_UNITS, config.INT16_UNITS,
                               config.INT32_UNITS, config.CATEGORY_MAX_SHARE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    typed.to_parquet(out_path, compression="zstd")
    log.info("typed: %s, %s, %.1f МБ, типов назначено: %d", out_path.name, typed.shape,
             out_path.stat().st_size / 1024**2, len(plan))


def run(stage, src_path, sample):
    t0 = time.perf_counter()
    p = paths(sample)
    todo = STAGES if stage == "all" else (stage,)
    if "serialize" in todo:
        stage_serialize(src_path, config.SHEETS_RAW_DIR)
    if "clean" in todo:
        stage_clean(config.SHEETS_RAW_DIR, p["untyped"], sample)
    if "typed" in todo:
        stage_typed(config.SHEETS_RAW_DIR, p["untyped"], p["final"])
    log.info("готово за %.1f с", time.perf_counter() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Обработка датасета GHS-UCDB")
    ap.add_argument("--input", type=Path, default=config.RAW_XLSX, help="исходный .xlsx")
    ap.add_argument("--stage", choices=("all",) + STAGES, default="all", help="этап (по умолчанию все)")
    ap.add_argument("--sample", type=int, default=None, metavar="N",
                    help="отладочный режим: первые N строк каждого листа, результат в data/interim/sample")
    ap.add_argument("--log-file", type=Path, default=config.LOGS_DIR / "cleaning.log")
    a = ap.parse_args(argv)
    setup_logging(a.log_file)
    try:
        run(a.stage, a.input, a.sample)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("необработанная ошибка")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
