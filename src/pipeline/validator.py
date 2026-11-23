"""Normalize OCR text and check it against Tunisian plate formats."""
import re

FORMATS = {
    "TUN": re.compile(r"^(\d{1,3})T(\d{1,4})$"),   # 123 تونس 4567
    "NT": re.compile(r"^(\d{1,6})N$"),             # 123456 ن.ت
}


def validate(text: str) -> dict:
    """Return the format, a status and a human-readable version of the plate."""
    text = text.strip().upper().replace(" ", "")
    if not text:
        return {"text": "", "format": None, "status": "empty", "display": ""}
    for name, pattern in FORMATS.items():
        m = pattern.match(text)
        if m:
            display = f"{m[1]} تونس {m[2]}" if name == "TUN" else f"{m[1]} ن.ت"
            return {"text": text, "format": name, "status": "valid", "display": display}
    return {"text": text, "format": None, "status": "invalid", "display": text}