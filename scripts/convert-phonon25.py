#!/usr/bin/env python3
"""Convert FermionResearch/Phonon-2.5-English (or -2.5) to a dense F32 GGUF.

transcribe.cpp's converter has the Fermion container reader baked to
Phonon-2 (repo, revision, archive hash). The container format is the same
`fermion-five-value-parakeet-v1`, so this driver reuses that reader with
the 2.5 artifact pinned, then hands off to upstream's `convert()` with a
2.5 profile. Run it once; the runtime never touches Python.

    python scripts/convert-phonon25.py [--variant en|multi] [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import zstandard
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
TP = REPO_ROOT / "native" / "third_party" / "transcribe.cpp"
CONVERTER = TP / "scripts" / "convert-parakeet.py"

VARIANTS = {
    "en": {
        "repo": "FermionResearch/Phonon-2.5-English",
        "revision": "da200358d638886bc96c758c672ad5568828b119",
        "archive": "phonon-2.5-en.bps.tar.zst",
        "archive_sha256": "6bd8119ac5a637e5ece0666f5c1e43dbcd99e1c36b3d0ef3ec5f9b58260a9dd4",
        "container_sha256": "c57119c0f04756932acc1d60d90506346e3708d69f00e654084acc7fe507fe27",
        "slug": "phonon-2.5-en",
        "profile": {
            "variant": "tdt-0.6b-phonon-2.5-en",
            "display_name": "Phonon-2.5 English",
            "version": "v2.5",
            "size_label": "0.6B",
            "head_kind": "tdt",
            "expected_vocab_size": 8192,
            "languages": ["en"],
            "lang_detect": False,
            "source_format": "fermion",
            "transformers_runtime": True,
            "author": "Fermion Research",
            "organization": "FermionResearch",
            "license": "cc-by-4.0",
            "license_name": "Creative Commons Attribution 4.0",
            "license_link": "https://creativecommons.org/licenses/by/4.0/",
        },
    },
    "multi": {
        "repo": "FermionResearch/Phonon-2.5",
        "revision": "main",
        "archive": "phonon-2.5.bps.tar.zst",
        "archive_sha256": "1faa04053a472cabab3bc31d07f30c23f22906f71044ca10eb1f0de36d1f19a9",
        "container_sha256": "17cb62d08e751a41819eb779db089aa7ac4eb2bab433ba5c2991d8cbd3cfefb9",
        "slug": "phonon-2.5",
        "profile": {
            "variant": "tdt-0.6b-phonon-2.5",
            "display_name": "Phonon-2.5",
            "version": "v2.5",
            "size_label": "0.6B",
            "head_kind": "tdt",
            "expected_vocab_size": 8192,
            "languages": [
                "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
                "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru",
                "sk", "sl", "es", "sv", "uk",
            ],
            "lang_detect": False,
            "source_format": "fermion",
            "transformers_runtime": True,
            "author": "Fermion Research",
            "organization": "FermionResearch",
            "license": "cc-by-4.0",
            "license_name": "Creative Commons Attribution 4.0",
            "license_link": "https://creativecommons.org/licenses/by/4.0/",
        },
    },
}


def verify_sha256(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA256 mismatch for {path}: {actual} != {expected}")


def fetch_bundle(spec: dict) -> Path:
    """Download + extract the pinned artifact once, into ~/.cache/hex-phonon."""
    root = Path.home() / ".cache" / "hex-phonon" / spec["slug"]
    if (root / "model.fermion").is_file():
        verify_sha256(root / "model.fermion", spec["container_sha256"])
        return root

    from huggingface_hub import hf_hub_download

    archive = Path(hf_hub_download(spec["repo"], spec["archive"], revision=spec["revision"]))
    verify_sha256(archive, spec["archive_sha256"])
    root.mkdir(parents=True, exist_ok=True)
    allowed = {"model.fermion", "config.json", "packed_manifest.json", "bps_manifest.json"}
    with archive.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if member.name not in allowed:
                    raise SystemExit(f"unexpected archive member: {member.name}")
                target = root / member.name
                with tar.extractfile(member) as src, target.open("wb") as out:
                    import shutil
                    shutil.copyfileobj(src, out)
    verify_sha256(root / "model.fermion", spec["container_sha256"])
    return root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="en")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    bundle = fetch_bundle(spec)

    sys.path.insert(0, str(TP / "scripts"))
    import lib.phonon as phonon

    # Point upstream's reader at the 2.5 artifact we just verified.
    phonon.REPO = spec["repo"]
    phonon.REVISION = spec["revision"]
    phonon.ARCHIVE = spec["archive"]
    phonon.ARCHIVE_SHA256 = spec["archive_sha256"]
    phonon.CONTAINER_SHA256 = spec["container_sha256"]
    phonon.source_directory = lambda model_spec, revision=None: bundle

    module_spec = importlib.util.spec_from_file_location("convert_parakeet", CONVERTER)
    conv = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(conv)

    conv.VARIANT_PROFILES[spec["slug"]] = spec["profile"]

    out = args.out or (REPO_ROOT / "native" / "models" / spec["slug"] / f"{spec['slug']}-F32.gguf")
    out.parent.mkdir(parents=True, exist_ok=True)
    conv.convert(spec["repo"], out, repo_id=spec["repo"], revision=spec["revision"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
