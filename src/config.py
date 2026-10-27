"""Пути и параметры проекта."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "logs"

RAW_XLSX = RAW_DIR / "GHS_UCDB_REGION_LATIN_AMERICA_AND_THE_CARIBBEAN_R2024A.xlsx"
SHEETS_RAW_DIR = INTERIM_DIR / "sheets_raw"
CLEAN_UNTYPED = INTERIM_DIR / "clean_untyped.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "clean.parquet"
TYPED_PARQUET = INTERIM_DIR / "clean_typed.parquet"

KEY = "ID_UC_G0"
BASE_SHEET = "GENERAL_CHARACTERISTICS"
SERVICE_SHEETS = ("Description", "Index")

# --- Схема типов (шаг 10): тип выбирается по единице измерения из листа Index ---
# Границы задаются предметной областью, а не текущей выборкой.
FLOAT32_UNITS = {
    "%", "ton/year", "ton/(year x person)", "°C", "°C/year", "years", "m2/person",
    "Mbps", "dimensionless", "-", "num hazards", "ms", "mm/year", "mm year-1",
    "hectares", "num devices", "US$", "num facilities", "num facilities/km2",
    "num facilities/person", "num events", "m/s", "kWh/kWp", "m/m2",
    "Number of premature deaths",
}
INT16_UNITS = {"Num. of days", "year", "µg/m3", "%"}
INT32_UNITS = {"km2", "hectares", "num events", "num fires", "num people"}
CATEGORY_MAX_SHARE = 0.5
