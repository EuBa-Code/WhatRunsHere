"""Pull the five benchmark figures a publisher reports out of a model card.

Usage: python tools/extract_benchmarks.py <workdir> [extras]

<workdir>/cards/index.json lists the models ({id, display_name, card, file})
and <workdir>/cards/<file> holds each card's markdown, as downloaded from
https://huggingface.co/<card>/raw/main/README.md. With `extras`,
<workdir>/bench/extra_pages.json maps a model id to [[url, path], ...] of
further pages to read (a publisher's blog post, the original card behind a
mirror). Output: <workdir>/bench/cards_extracted.json.

Tables come in two orientations (benchmarks as rows and models as columns, or
the reverse), in Markdown or HTML. The model's own column or row is found by
name, with an exact name preferred over a sibling that merely contains it,
and the benchmark by a small vocabulary; the first number in the cell is the
score. Nothing is estimated: a model whose card has no table gets nothing.
The Open LLM Leaderboard fallback and the merge into sources.json are
separate passes, not part of this script.
"""
import json, re, sys, html

S = sys.argv[1]
idx = json.load(open(f"{S}/cards/index.json", encoding="utf-8"))
extra = json.load(open(f"{S}/bench/extra_pages.json", encoding="utf-8")) if len(sys.argv) > 2 else {}

METRICS = {
    "mmlu_pro": lambda c: ("mmlu-pro" in c or "mmlu pro" in c or "mmlu_pro" in c) and "redux" not in c,
    "gpqa_diamond": lambda c: "gpqa" in c,
    "livecodebench": lambda c: "livecodebench" in c or "live code bench" in c,
    "math": lambda c: (("math-500" in c or "math500" in c or "math 500" in c)
                       or (re.fullmatch(r"\**math\**( \(.*\))?", c.strip()) is not None))
                      and not any(x in c for x in ("gsm", "aime", "vista", "lvl", "arena", "hmmt", "verse")),
    "ifeval": lambda c: "ifeval" in c and "multi" not in c,
}
NUM = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%?")

def clean(cell):
    cell = html.unescape(re.sub(r"<[^>]+>", " ", cell))
    return re.sub(r"[*_`]", "", cell).strip()

def number(cell):
    m = NUM.search(clean(cell))
    if not m:
        return None
    v = float(m.group(1))
    if 0 < v <= 1.0 and "." in m.group(1):
        v *= 100
    return v if 0 <= v <= 100 else None

def md_tables(text):
    tables, block = [], []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            block.append(line)
        elif block:
            tables.append(block); block = []
    if block:
        tables.append(block)
    out = []
    for block in tables:
        rows = []
        for line in block:
            cells = [c for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r"\s*:?-{2,}:?\s*", c) for c in cells if c.strip()):
                continue
            rows.append([clean(c) for c in cells])
        if len(rows) >= 2:
            out.append(rows)
    return out

def html_tables(text):
    out = []
    for t in re.findall(r"<table.*?</table>", text, flags=re.S | re.I):
        rows = []
        for tr in re.findall(r"<tr.*?</tr>", t, flags=re.S | re.I):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S | re.I)
            if cells:
                rows.append([clean(c) for c in cells])
        if len(rows) >= 2:
            out.append(rows)
    return out

def squash(text):
    """Lowercase alphanumerics only, so `InternLM2.5-20B-Chat`, `internlm2_5-20b-chat`
    and `InternLM 2.5 20B Chat` are one name."""
    return re.sub(r"[^a-z0-9]", "", text.lower())

SIZE = re.compile(r"\d+(?:\.\d+)?b")

def names_for(model_id, display):
    tail = model_id.split("/")[-1]
    names = [tail, display, re.sub(r"-\d{4}(?=-)", "", tail)]
    for suffix in ("-Instruct-HF", "-Instruct", "-instruct", "-it", "-Chat", "-chat", "-HF", "-PT"):
        for n in list(names):
            if n and n.endswith(suffix):
                names.append(n[: -len(suffix)])
    names = [squash(n) for n in names if n]
    return sorted(set(n for n in names if len(n) >= 5), key=len, reverse=True)

def model_score(cell, names):
    """2 for the model's own name, 1 for a cell that contains it or abbreviates
    it with the same size token, 0 otherwise. A comparison table lists the
    model beside its siblings and its base, so exact beats contained."""
    c = squash(cell)
    if len(c) < 5:
        return 0
    if c in names:
        return 2
    sizes = set(SIZE.findall(names[0]))
    for n in names:
        if (n in c or (len(c) >= 9 and c in n)) and sizes <= set(SIZE.findall(c)):
            return 1
    return 0

def cell_is_model(cell, names):
    return model_score(cell, names) > 0

def metric_of(cell):
    c = cell.lower()
    for key, test in METRICS.items():
        if test(c):
            return key
    return None

def extract(rows, names):
    found = {}
    header = rows[0]
    # Benchmarks across the header, models down the first column: the row
    # that names the model best.
    if any(metric_of(h) for h in header[1:]):
        scored = [(model_score(row[0], names), i) for i, row in enumerate(rows[1:], 1) if row]
        best = max(scored, default=(0, None))
        if best[0] > 0:
            row = rows[best[1]]
            for h, cell in zip(header[1:], row[1:]):
                m = metric_of(h)
                if m and m not in found and number(cell) is not None:
                    found[m] = number(cell)
            if found:
                return found
    # Models across the header, benchmarks down the first column. The header
    # may span two rows (a category row above the model row).
    for hdr_i in (0, 1):
        if hdr_i >= len(rows):
            break
        hdr = rows[hdr_i]
        scored = [(model_score(h, names), i) for i, h in enumerate(hdr) if i > 0]
        best = max(scored, default=(0, None))
        if best[0] == 0:
            continue
        col = best[1]
        for row in rows[hdr_i + 1:]:
            if not row:
                continue
            m = metric_of(row[0]) or (metric_of(row[1]) if len(row) > 2 and not number(row[1]) else None)
            if m and m not in found and col < len(row) and number(row[col]) is not None:
                found[m] = number(row[col])
        if found:
            return found
    return found

results = []
for entry in idx:
    mid, display = entry["id"], entry["display_name"]
    texts = []
    if entry.get("file"):
        texts.append((f"https://huggingface.co/{entry['card']}", open(f"{S}/cards/{entry['file']}", encoding="utf-8").read()))
    for url, path in extra.get(mid, []):
        texts.append((url, open(path, encoding="utf-8").read()))
    names = names_for(mid, display)
    found, source = {}, None
    for url, text in texts:
        for table in md_tables(text) + html_tables(text):
            got = extract(table, names)
            for k, v in got.items():
                if k not in found:
                    found[k] = v
                    source = source or url
        if len(found) >= 4:
            break
    results.append({"id": mid, **{k: found.get(k) for k in METRICS}, "source": source})
    flags = "".join(k[0].upper() if found.get(k) is not None else "." for k in METRICS)
    vals = " ".join(f"{k[:4]}={found[k]:.1f}" for k in METRICS if found.get(k) is not None)
    print(f"{flags} {mid:46s} {vals}")

json.dump(results, open(f"{S}/bench/cards_extracted.json", "w", encoding="utf-8"), indent=1)
print("with any figure:", sum(1 for r in results if any(r[k] is not None for k in METRICS)), "of", len(results))
