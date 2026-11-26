"""Download English Wikipedia articles as clean text for the Valley RAG."""
import csv
import json
import re
import time
from datetime import date
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "PlateRecoValley/1.0 (student project; contact: your_email@example.com)"}
RAW = Path("data/rag/raw")
SOURCES = Path("data/rag/sources.csv")
MIN_CHARS = 2000  # skip stubs (too short to be useful)

TOPICS = {
    "cars": [
        "Car", "Internal combustion engine", "Diesel engine", "Petrol engine",
        "Electric vehicle", "Hybrid vehicle", "Automatic transmission", "Manual transmission",
        "Disc brake", "Anti-lock braking system", "Electronic stability control",
        "Turbocharger", "Tire", "Airbag", "Seat belt", "Fuel economy in automobiles",
        "Vehicle identification number", "Vehicle registration plates of Tunisia",
    ],
    "maintenance": [
        "Motor oil", "Oil filter", "Spark plug", "Automotive battery", "Brake pad",
        "Antifreeze", "Timing belt (camshaft)", "Check engine light",
        "Tire rotation", "Wheel alignment", "Air filter", "Car wash",
    ],
}


def fetch(title: str) -> dict | None:
    params = {"action": "query", "prop": "extracts|info", "explaintext": 1, "inprop": "url",
              "redirects": 1, "titles": title, "format": "json"}
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    if "missing" in page:
        return None
    return {"title": page["title"], "url": page["fullurl"], "text": page.get("extract", "")}


def clean(text: str) -> str:
    """Remove empty sections and the 'See also / References' tail."""
    text = re.split(r"\n=+ (See also|References|External links|Further reading|Notes) =+\n", text)[0]
    text = re.sub(r"\n=+ [^=]+ =+\n(?=\s*\n=+ )", "\n", text)   # headings with no content
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    known = set()
    if SOURCES.exists():
        with SOURCES.open(encoding="utf-8") as f:
            known = {row["file"] for row in csv.DictReader(f)}

    new_rows, missing, short = [], [], []
    for category, titles in TOPICS.items():
        (RAW / category).mkdir(parents=True, exist_ok=True)
        for title in titles:
            page = fetch(title)
            time.sleep(0.5)  # be polite with the Wikipedia API
            if page is None:
                missing.append(title)
                continue
            text = clean(page["text"])
            if len(text) < MIN_CHARS:
                short.append(f"{title} ({len(text)} chars)")
                continue

            slug = re.sub(r"[^a-z0-9]+", "_", page["title"].lower()).strip("_")
            rel = f"{category}/wiki_{slug}.json"
            doc = {"title": page["title"], "url": page["url"], "text": text,
                   "language": "en", "license": "CC BY-SA 4.0",
                   "fetched": date.today().isoformat()}
            (RAW / rel).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"OK  {rel:<55} {len(text):>7} chars")
            if rel not in known:
                new_rows.append([rel, category, page["title"], page["url"], "en",
                                 doc["fetched"], "CC BY-SA 4.0"])

    with SOURCES.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(new_rows)

    print(f"\n{len(new_rows)} new sources added to {SOURCES}")
    if missing:
        print("Not found:", missing)
    if short:
        print("Skipped (too short):", short)


if __name__ == "__main__":
    main()