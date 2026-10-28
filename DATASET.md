# Паспорт набора данных GHS-UCDB R2024A (Латинская Америка и Карибы)

## 1. Общие сведения

Источник: лист Description исходной книги (сокращено; тексты Disclaimer и условий использования приведены там).

- **Dataset ID**: GHS-UCDB R2024A
- **Version**: V1.2
- **Data Last Update**: 15/05/2026
- **Alternative name (with version)**: GHS Urban Centre Database 2024 - GHS Urban Centre Database 2025, Version 1.2
- **Authors**: Mari Rivero, Ines; Melchiorri, Michele; Florio, Pietro; Schiavina, Marcello; Goch, Katarzyna; Politis, Panagiotis; Uhl, Johannes H; Pesaresi, Martino; Maffenini, Luca; Sulis, Patrizia; Crippa, Monica; Guizzardi, Diego; Pisoni, Enrico; Belis, Claudio; Jacome Felix Oom, Duarte; Branco, Alfredo; Mwaniki, Dennis; Kochulem, Edwin; Githira, Daniel; Carioli, Alessandra; Ehrlich, Daniele; Tommasi, Pierpaolo; Kemper, Thomas; Dijkstra, Lewis
- **Document Version**: V1.1
- **Document Last Update**: 30/07/2025
- **Contact point**: JRC-GHSL<at>EC.EUROPA.EU
- **Description**: This dataset contains statistics on urban centres based on data from the Global Human Settlement Layer (GHSL) produced at the Joint Research Centre of the European Commission, unit E.1 (Disaster Risk Management). This release is based on the GHSL Data Package 2023, the Degree of Urbanisation to delineate spatial entities, and geospatial data integration from a variety of open source datasets to characterise them. The result is the most complete information system on cities to date with data for 11,422 quality-controlled urban centres across 15 thematic domains, 471 indicators, and 2600 attributes. The UCDB has two data streams, one based on a fixed delineation of urban centres in 2025, and a second version based on multi-temporal delineation of urban centres, traceable over time. The UCDB integrates data from Copernicus Services (including Emergency, Land Monitoring, Marine and Climate), peer-reviewed datasets (i.e. from the scientific literature), and institutional information systems (i.e. from the United Nations).
- **Licence**: European Commission Reuse and Copyright Notice
- **Dataset Citation**: Mari Rivero, Ines; Melchiorri, Michele; Florio, Pietro; Schiavina, Marcello; Goch, Katarzyna; Politis, Panagiotis; Uhl, Johannes H; Pesaresi, Martino; Maffenini, Luca; Sulis, Patrizia; Crippa, Monica; Guizzardi, Diego; Pisoni, Enrico; Belis, Claudio; Jacome Felix Oom, Duarte; Branco, Alfredo; Mwaniki, Dennis; Kochulem, Edwin; Githira, Daniel; Carioli, Alessandra; Ehrlich, Daniele; Tommasi, Pierpaolo; Kemper, Thomas; Dijkstra, Lewis (2024): GHS-UCDB R2024A - GHS Urban Centre Database 2025. European Commission, Joint Research Centre (JRC) [Dataset] doi: 10.2905/1a338be6-7eaf-480c-9664-3a8ade88cbcd PID: http://data.europa.eu/89h/1a338be6-7eaf-480c-9664-3a8ade88cbcd
- **Related Resources**: Stats in the City – the GHSL Urban Centre Database 2025. MELCHIORRI, M., MARI RIVERO, I., FLORIO, P., SCHIAVINA, M., KRASNODEBSKA, K., POLITIS, P., UHL, J., PESARESI, M., MAFFENINI, L., SULIS, P., CRIPPA, M., GUIZZARDI, D., PISONI, E., BELIS, C., OOM, D., BRANCO, A., MWANIKI, D., GITHIRA, D., KOCHULEM, E., TOMMASI, P., CARIOLI, A., EHRLICH, D., KEMPER, T. and DIJKSTRA, L., Stats in the City – the GHSL Urban Centre Database 2025, Publications Office of the European Union, Luxembourg, 2024, doi:10.2760/3046391 (online),10.2760/5259274 (print), JRC139768.

Ключ: `ID_UC_G0` (1089 городских центров). Файлы: `.xlsx` (17 листов: 15 тематических, Description, Index), `.gpkg` (15 слоёв и UC_centroids), два PDF с документацией.

## 2. Структура

После очистки: 1089 строк x 2619 столбцов (`data/processed/clean.parquet`, Parquet zstd, с назначенными типами).

| Лист | Столбцов в источнике | Столбцов в итоговой таблице |
|---|---|---|
| GENERAL_CHARACTERISTICS | 13 | 12 |
| GHSL | 551 | 494 |
| CLIMATE | 109 | 102 |
| EMISSIONS | 590 | 583 |
| EXPOSURE | 1066 | 984 |
| GEOGRAPHY | 37 | 23 |
| GREENNESS | 63 | 56 |
| HAZARD_RISK | 118 | 76 |
| HEALTH | 17 | 10 |
| INFRASTRUCTURES | 17 | 10 |
| LULC | 49 | 30 |
| NATURAL_SYSTEMS | 23 | 14 |
| SDG | 43 | 36 |
| SOCIOECONOMIC | 183 | 154 |
| WATER | 28 | 21 |

Примечание: столбцы `GC_*` повторяются на нескольких листах источника, а в таблице каждый учтён один раз, поэтому числа по листам приблизительные.

## 3. Словарь столбцов

Полный словарь: `docs/columns_dictionary.csv` (имя, лист, тематическая область, группа, показатель, единица, итоговый тип, доля пропусков, способ сопоставления с Index).

Сопоставление с листом Index (шаблоны вида `GC_CNT_GAD_XXXX`, где `XXXX` заменяет год):
- всего столбцов: 2619;
- сопоставлено с Index: 2537, из них по нормализованному имени (расхождения в записи имён): 128, по предполагаемому имени: 23;
- служебные столбцы, добавленные очисткой: 13 (`dup_key_flag`, `*_cens`, `out_of_range`, `*_out`).

### Столбцы без описания в Index

- GEOGRAPHY: `ge_mrb_lis` (1 столбец)
- GEOGRAPHY: `ge_mrb_mai` (1 столбец)
- GHSL: `gh_xst_d11` (10 столбцов)
- GHSL: `gh_xst_d12` (10 столбцов)
- GHSL: `gh_xst_d13` (8 столбцов)
- GHSL: `gh_xst_d21` (10 столбцов)
- GHSL: `gh_xst_d22` (8 столбцов)
- GHSL: `gh_xst_d23` (10 столбцов)
- GHSL: `gh_xst_d30` (11 столбцов)

Смысл этих столбцов в паспорте не указан: его нужно уточнять по PDF-документации.

### Дефекты заголовков в источнике

- `XST_A23_*`, `XST_A30_*` (21 столбец): в заголовке нет префикса `GH_`, в Index записано `GH_XST_A23_XXXX`, `GH_XST_A30_XXXX`.
- `GH_L30_2025`, `GH_W30_2025`: в Index записано `GH_XST_L30_XXXX`, `GH_XST_W30_XXXX`. Сопоставление предположительное.
- Расхождения в записи имён между Index и таблицей: `EX_01_POP` и `ex_e01_pop`, `EX_020_S1N` и `ex_20_s1n`, `GC_LCZ_*` и `cl_lcz_*`.
- Обрезанное имя `NS_TPA_DES_202` исправлено на `NS_TPA_DES_2025`.

## 4. Особенности данных

- Кодировка: 918 испорченных ячеек исправлено (0 после очистки).
- NATURAL_SYSTEMS: 2 пары дублей по ключу (4943 Cúcuta, 6621 Reynosa), оставлена первая, флаг `dup_key_flag`.
- EXPOSURE: пропуски структурные (414 городов, только Upper Middle income), NaN означает «не посчитано», не заполняется.
- EM_PM2_MOR_2000..2020: числа хранились текстом, приведены к float.
- SDG: пределы '>5' и '<-5' (664 значения) заменены на +-5, флаги `*_cens`.
- Доли больше 100 %: 48 столбцов, 870 значений, не обрезаны, флаг `out_of_range`.
- Выбросы (IQR на log1p): строки не удаляются, флаги `*_out`.
- Списки через `;` (NS_TPA_NAM_2025, NS_TPA_DES_2025, NS_MPA_*) нарушают 1НФ.
- GH_POP_CAG_*: отрицательные значения допустимы (темп прироста).
- Удалены 171 константный и 29 разреженных столбцов (`logs/dropped_*_columns.csv`).
- 241 числовой столбец без единицы в Index.
