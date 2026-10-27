"""Функции очистки данных GHS-UCDB."""


def fix_mojibake(s):
    """Чинит только строки с признаками двойного кодирования UTF-8."""
    if not isinstance(s, str) or not any(m in s for m in ("Ã", "Â")):
        return s
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s
