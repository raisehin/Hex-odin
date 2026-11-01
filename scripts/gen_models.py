#!/usr/bin/env python3
"""Regenerate the model registry in src/models.odin from Handy's catalog.

Handy keeps every downloadable model in

    src-tauri/src/catalog/catalog.json   (schema 2, cjpais/Handy)

with repository, revision, per-quantization filename/size/sha256, languages,
capabilities and benchmark results. Hex used a hand-written Rust registry that
went stale; this script makes the Odin registry a projection of that catalog, so
a refresh is one command:

    git -C <handy> pull
    python scripts/gen_models.py <handy>/src-tauri/src/catalog/catalog.json

Only role == "asr" entries are written (diarization, language-ID and VAD models
are not dictation models), and KEEP_IDS trims the catalog down to the two
models this build ships: Phonon-2.5 English and Canary 180M Flash. The Q8_0
file is used, matching `quant` in the catalog and Handy's own default.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "models.odin"
KEEP_FROM = "model_by_id :: proc"
EXTRA = ROOT / "scripts" / "models_extra.json"

# Deliberately short menu. Anything not listed here is dropped even if the
# catalog carries it, so `hex models list` never shows a model nobody chose.
KEEP_IDS = {"phonon25_en", "canary-180m-flash"}

HEADER = '''package main

import "core:fmt"
import "core:os"
import "core:strings"

// GENERATED from cjpais/Handy src-tauri/src/catalog/catalog.json (schema 2).
// Regenerate with scripts/gen_models.py; do not hand-edit the table.
// Adding a model is data: transcribe.cpp reads the architecture out of the GGUF.
Model :: struct {
\tid:          string,
\tname:        string,
\tfamily:      string,
\tdescription: string,
\trepo:        string,
\tfilename:    string,
\trevision:    string,
\tbytes:       u64,
\tsha256:      string,
\tlanguages:   string,
\trealtime:    string,
\tstreams:     bool,
\tcurated:     bool,
}

MODELS := []Model{
'''


def lang_label(model):
    langs = model.get("languages") or []
    if not langs:
        return "?"
    if len(langs) == 1:
        return langs[0]
    return f"{len(langs)} languages"


def quant_file(model):
    files = model["files"]
    want = model.get("quant")
    return files.get(want) or files.get("Q8_0") or list(files.values())[0]


def speed(model, results):
    perf = (results.get(model["id"]) or {}).get("performance") or {}
    value = perf.get("accelerated")
    if value is None:
        value = perf.get("cpu")
    return "-" if value is None else f"~{value:.0f}x"


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    catalog = json.loads(pathlib.Path(argv[1]).read_text(encoding="utf-8"))
    models = [m for m in catalog["models"] if m.get("role") == "asr"]
    results = catalog.get("results") or {}
    curated = {m["id"] for m in catalog["models"] if m.get("curated")}
    # Curated entries first in catalog order (the catalog is ranked), then the
    # uncurated tail alphabetically, then local extras (models the catalog does
    # not carry, e.g. Phonon-2 GGUFs published outside Handy).
    order = {m["id"]: i for i, m in enumerate(catalog["models"])}
    models = [m for m in models if m["id"] in KEEP_IDS]
    models.sort(key=lambda m: (m["id"] not in curated, order[m["id"]] if m["id"] in curated else 0, m["name"].lower()))
    extras = json.loads(EXTRA.read_text(encoding="utf-8")) if EXTRA.exists() else []
    seen = {m["id"] for m in models}
    extras = [e for e in extras if e["id"] not in seen and e["id"] in KEEP_IDS]

    out = [HEADER]
    for e in extras:
        out.append(
            "\t{\n"
            f'\t\tid          = "{e["id"]}",\n'
            f'\t\tname        = "{e["name"]}",\n'
            f'\t\tfamily      = "{e.get("family", "")}",\n'
            f'\t\tdescription = "{e.get("description", "")}",\n'
            f'\t\trepo        = "{e["repo"]}",\n'
            f'\t\tfilename    = "{e["filename"]}",\n'
            f'\t\trevision    = "{e["revision"]}",\n'
            f'\t\tbytes       = {e["bytes"]},\n'
            f'\t\tsha256      = "{e["sha256"]}",\n'
            f'\t\tlanguages   = "{e.get("languages", "?")}",\n'
            f'\t\trealtime    = "{e.get("realtime", "-")}",\n'
            f'\t\tstreams     = {"true" if e.get("streams") else "false"},\n'
            f'\t\tcurated     = {"true" if e.get("curated", True) else "false"},\n'
            "\t},\n"
        )
    for m in models:
        f = quant_file(m)
        desc = (m.get("description") or "").replace('"', '\\"')
        streams = "true" if (m.get("capabilities") or {}).get("streaming") else "false"
        is_curated = "true" if m["id"] in curated else "false"
        out.append(
            "\t{\n"
            f'\t\tid          = "{m["id"]}",\n'
            f'\t\tname        = "{m["name"]}",\n'
            f'\t\tfamily      = "{m.get("family", "")}",\n'
            f'\t\tdescription = "{desc}",\n'
            f'\t\trepo        = "{m["repository"]}",\n'
            f'\t\tfilename    = "{f["filename"]}",\n'
            f'\t\trevision    = "{m["revision"]}",\n'
            f'\t\tbytes       = {f["size"]},\n'
            f'\t\tsha256      = "{f["sha256"]}",\n'
            f'\t\tlanguages   = "{lang_label(m)}",\n'
            f'\t\trealtime    = "{speed(m, results)}",\n'
            f"\t\tstreams     = {streams},\n"
            f"\t\tcurated     = {is_curated},\n"
            "\t},\n"
        )
    out.append("}\n\n")

    existing = TARGET.read_text(encoding="utf-8")
    out.append(existing[existing.index(KEEP_FROM):])
    TARGET.write_text("".join(out), encoding="utf-8")
    print(f"wrote {len(models) + len(extras)} ASR models ({len(curated & {m['id'] for m in models})} curated) to {TARGET}")


if __name__ == "__main__":
    main(sys.argv)
