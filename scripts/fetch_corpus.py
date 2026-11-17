"""Fetch the raw corpus from Wikipedia (CC BY-SA 4.0) and write it under data/raw/.

Two sources:
  1. "Provinces of Thailand" -> the statistics table (population Dec-2024, area, density,
     capital, ISO code) -> data/raw/provinces_table.csv
  2. "Regions of Thailand"   -> the six-region (geographical) grouping, joined onto (1)
  3. One article per province -> data/raw/provinces/<slug>.md (plain text)

The output is committed to the repo, so you do NOT need to run this to use the app.
It is here so the corpus is reproducible:   python scripts/fetch_corpus.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "thai-province-agent-capstone/0.1 (educational project)"}


def _get(params: dict) -> dict:
    params = {"format": "json", "formatversion": 2, **params}
    for attempt in range(6):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        # 429 = rate limited: honour Retry-After, else back off exponentially.
        wait = int(r.headers.get("Retry-After", 0) or 0) or 5 * 2**attempt
        print(f"  HTTP {r.status_code}, retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    r.raise_for_status()
    return {}


def wikitext(page: str) -> str:
    return _get({"action": "parse", "page": page, "prop": "wikitext"})["parse"]["wikitext"]


def norm(name: str) -> str:
    """'Buri Ram' / 'Buriram' / 'Si Sa Ket' / 'Sisaket' -> one key."""
    return re.sub(r"[^a-z]", "", name.lower())


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _int(cell: str) -> int:
    cell = cell.split("|")[-1]
    return int(re.sub(r"[^\d]", "", cell))


def parse_province_table(text: str) -> list[dict]:
    start = text.find('{| class="wikitable sortable"')
    end = text.find("|}", start)
    rows = text[start:end].split("\n|-")[1:]
    out = []
    for row in rows:
        cells = [c.strip() for c in row.strip().split("\n|") if c.strip()]
        cells = [c.lstrip("| ").strip() for c in cells]
        if len(cells) < 9 or "{{flag|" not in row:
            continue
        name = re.search(r"\{\{flag\|([^}|]+)", row).group(1).strip()
        thai = re.search(r"\{\{lang\|th\|([^}]+)\}\}", row).group(1).strip()
        nums = [c for c in cells if re.search(r"text-align:right", c)]
        capital = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", cells[6])
        out.append(
            {
                "name": name,
                "name_thai": thai,
                "population": _int(nums[0]),
                "area_km2": _int(nums[1]),
                "density_per_km2": _int(nums[2]),
                "capital": (capital.group(2) or capital.group(1)).strip() if capital else cells[6],
                "iso_code": next(c for c in cells if c.startswith("TH-")),
            }
        )
    return out


def parse_regions(text: str) -> dict[str, str]:
    """Return {norm(province): six-region name} from the Comparison table."""
    start = text.find("=== Comparison ===")
    end = text.find("|}", start)
    regions: dict[str, str] = {}
    carry, carry_left = "", 0  # the six-region cell may span rows (rowspan="N")
    for row in text[start:end].split("\n|-")[1:]:
        lines = [l for l in row.strip().split("\n") if l.startswith("|")]
        if not lines:
            continue
        names = re.findall(r"\[\[[^\]|]+\|([^\]]+)\]\]", lines[0])
        if carry_left > 0:
            region, carry_left = carry, carry_left - 1
        else:
            cell = lines[1].lstrip("|")
            region = cell.split("|")[-1].strip()
            span = re.search(r'rowspan="(\d+)"', cell)
            if span:
                carry, carry_left = region, int(span.group(1)) - 1
        for n in names:
            regions[norm(n)] = region
    return regions


ALIASES = {  # spellings that differ between the two Wikipedia tables / article titles
    "buriram": "buriram", "sisaket": "sisaket", "lopburi": "lopburi",
}


def article_title(name: str) -> str:
    return "Bangkok" if name == "Bangkok" else f"{name} province"


def fetch_article(title: str) -> tuple[str, str]:
    data = _get(
        {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": 1,
            "redirects": 1,
            "inprop": "url",
            "titles": title,
        }
    )
    page = data["query"]["pages"][0]
    return page.get("extract", ""), page.get("fullurl", "")


def clean_extract(text: str) -> str:
    # Drop trailing reference-ish sections; they add noise but no answers.
    text = re.split(r"\n==+ (See also|References|Notes|External links|Further reading|Bibliography) ==+", text)[0]
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Turn "== Heading ==" into markdown headings so the chunker can use them.
    text = re.sub(r"^==== (.+?) ====$", r"#### \1", text, flags=re.M)
    text = re.sub(r"^=== (.+?) ===$", r"### \1", text, flags=re.M)
    text = re.sub(r"^== (.+?) ==$", r"## \1", text, flags=re.M)
    return text.strip()


def main() -> int:
    (RAW / "provinces").mkdir(parents=True, exist_ok=True)

    table = parse_province_table(wikitext("Provinces of Thailand"))
    regions = parse_regions(wikitext("Regions of Thailand"))
    regions[norm("Bangkok")] = "Central"
    missing = []
    for row in table:
        row["region"] = regions.get(norm(row["name"]), "")
        if not row["region"]:
            missing.append(row["name"])
    if missing:
        print("WARNING: no region for", missing, file=sys.stderr)
    print(f"parsed {len(table)} provinces")

    manifest = []
    for row in table:
        title = article_title(row["name"])
        doc_id = slug(row["name"])
        row["doc_id"] = doc_id
        path = RAW / "provinces" / f"{doc_id}.md"
        if path.exists() and "--force" not in sys.argv:  # resume an interrupted fetch
            existing = path.read_text(encoding="utf-8")
            url = re.search(r"Source: (\S+)", existing).group(1)
            body = existing.split("\n\n", 2)[2]
            manifest.append({"doc_id": doc_id, "title": row["name"], "url": url, "chars": len(body)})
            continue
        text, url = fetch_article(title)
        if not text:
            print("WARNING: empty article for", title, file=sys.stderr)
            continue
        body = clean_extract(text)
        path.write_text(
            f"# {row['name']}\n\nSource: {url} (Wikipedia, CC BY-SA 4.0)\n\n{body}\n", encoding="utf-8"
        )
        manifest.append({"doc_id": doc_id, "title": row["name"], "url": url, "chars": len(body)})
        print(f"  {doc_id:28s} {len(body):>7,} chars")
        time.sleep(1.5)

    fields = ["doc_id", "name", "name_thai", "region", "capital", "population",
              "area_km2", "density_per_km2", "iso_code"]
    with open(RAW / "provinces_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in table:
            w.writerow({k: row.get(k, "") for k in fields})
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(manifest)} articles + provinces_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
