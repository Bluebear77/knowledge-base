#!/usr/bin/env python3
"""
Extract all entities from a DOREMUS RDF dump and write them into one JSON file
in a SPARQL-results-like format:

{
  "head": { "link": [], "vars": ["id", "main_label"] },
  "results": {
    "distinct": false,
    "ordered": true,
    "bindings": [
      {
        "id": {"type": "uri", "value": "..."},
        "main_label": {"type": "literal", "value": "..."}
      }
    ]
  }
}

What this script does
---------------------
1. Recursively scans a dump directory for RDF files.
2. Parses files one by one with rdflib.
3. Extracts every URI entity encountered as a subject or object.
4. Tries to find the best human-readable label for each entity.
5. Falls back to a label derived from the URI when no explicit label exists.
6. Writes one final JSON file.

Why this design
---------------
- The DOREMUS dump can be large, so the script parses one file at a time.
- It does NOT keep a full RDF graph of the whole dump in memory.
- It still keeps a Python dictionary of extracted entities in memory, because
  the final output must be a single JSON object.

Dependencies
------------
pip install rdflib tqdm

Example
-------
python extract_doremus_entities.py \
    --input ~/Documents/EURECOM/PhD2/doremus_dump \
    --output doremus_entities.json

Notes
-----
- The script prefers labels in this order:
    skos:prefLabel > rdfs:label > foaf:name > dc:title > dcterms:title > schema:name
- Language preference:
    no language > en > fr > any other language
- If an entity has no explicit label, a fallback label is derived from the URI
  fragment or the last path segment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from rdflib import Graph, Literal, URIRef
from tqdm import tqdm


# ------------------------------------------------------------------------------
# RDF namespaces / predicates we want to treat as possible labels.
# We store them as strings to keep comparisons cheap and simple.
# ------------------------------------------------------------------------------
LABEL_PREDICATE_PRIORITY = {
    # Higher priority = better label source
    "http://www.w3.org/2004/02/skos/core#prefLabel": 100,
    "http://www.w3.org/2000/01/rdf-schema#label": 90,
    "http://xmlns.com/foaf/0.1/name": 80,
    "http://purl.org/dc/elements/1.1/title": 70,
    "http://purl.org/dc/terms/title": 60,
    "http://schema.org/name": 50,
}


# ------------------------------------------------------------------------------
# File extensions that rdflib can usually parse.
# We map suffix -> rdflib format.
#
# Important:
# - ".ttl"      => Turtle
# - ".nt"       => N-Triples
# - ".nq"       => N-Quads
# - ".rdf/.xml" => RDF/XML
# - ".trig"     => TriG
# - ".jsonld"   => JSON-LD
# ------------------------------------------------------------------------------
FORMAT_BY_SUFFIX = {
    ".ttl": "turtle",
    ".nt": "nt",
    ".nq": "nquads",
    ".rdf": "xml",
    ".xml": "xml",
    ".trig": "trig",
    ".jsonld": "json-ld",
}


# ------------------------------------------------------------------------------
# Some filenames in dumps are clearly not RDF data and should be ignored.
# ------------------------------------------------------------------------------
IGNORED_FILENAMES = {
    "README",
    "README.md",
    "README.txt",
}


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def discover_rdf_files(root: Path) -> List[Tuple[Path, str]]:
    """
    Recursively discover RDF files under `root`.

    Returns
    -------
    list of (path, rdflib_format)
        Example:
        [
            (Path("organization.ttl"), "turtle"),
            (Path("something.nt"), "nt"),
        ]
    """
    rdf_files: List[Tuple[Path, str]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.name in IGNORED_FILENAMES:
            continue

        fmt = FORMAT_BY_SUFFIX.get(path.suffix.lower())
        if fmt is not None:
            rdf_files.append((path, fmt))

    # Sort for deterministic processing order
    rdf_files.sort(key=lambda x: str(x[0]))
    return rdf_files


def clean_whitespace(text: str) -> str:
    """
    Normalize whitespace in labels.
    """
    return re.sub(r"\s+", " ", text).strip()


def uri_to_fallback_label(uri: str) -> str:
    """
    Build a readable fallback label from a URI when no explicit label exists.

    Strategy:
    1. Use fragment after '#' if present.
    2. Otherwise use last path segment.
    3. URL-decode it.
    4. Replace underscores/hyphens with spaces.
    5. Trim whitespace.
    """
    parsed = urlparse(uri)

    # Prefer fragment if available, e.g. http://example.org#MyEntity -> MyEntity
    if parsed.fragment:
        raw = parsed.fragment
    else:
        # Otherwise use the final path segment
        path = parsed.path.rstrip("/")
        raw = path.split("/")[-1] if path else uri

    raw = unquote(raw)
    raw = raw.replace("_", " ").replace("-", " ")
    raw = clean_whitespace(raw)

    return raw if raw else uri


def language_score(lang: Optional[str]) -> int:
    """
    Score language preference.

    Higher is better.

    Preference chosen here:
    - no language tag      => 30
    - English ("en")       => 20
    - French ("fr")        => 10
    - any other language   => 0
    """
    if not lang:
        return 30

    lang = lang.lower()
    if lang == "en" or lang.startswith("en-"):
        return 20
    if lang == "fr" or lang.startswith("fr-"):
        return 10
    return 0


def candidate_label_score(predicate_uri: str, literal: Literal) -> Tuple[int, int, int]:
    """
    Compute a score tuple for a candidate label.

    Score order:
    1. Label predicate priority (prefLabel better than label, etc.)
    2. Language preference
    3. Slight preference for shorter labels (inverted via negative length)

    The returned tuple can be compared directly with ">".
    """
    pred_score = LABEL_PREDICATE_PRIORITY.get(predicate_uri, -1)
    lang_score = language_score(getattr(literal, "language", None))
    length_score = -len(str(literal))
    return (pred_score, lang_score, length_score)


# ------------------------------------------------------------------------------
# Core extraction logic
# ------------------------------------------------------------------------------


def extract_entities(root: Path) -> Dict[str, Dict[str, object]]:
    """
    Extract entities and their best label from all RDF files under `root`.

    Returns
    -------
    dict:
        {
            entity_uri: {
                "label": "Best label chosen so far",
                "score": (predicate_score, lang_score, -length)
            },
            ...
        }

    Notes
    -----
    - Every URIRef seen as subject or object is included as an entity.
    - If no explicit label is found, the label stays None until the end,
      when we fill it with a fallback derived from the URI.
    """
    rdf_files = discover_rdf_files(root)

    if not rdf_files:
        raise FileNotFoundError(f"No RDF files found under: {root}")

    entities: Dict[str, Dict[str, object]] = {}

    # Outer progress bar: one step per RDF file
    for file_path, rdf_format in tqdm(rdf_files, desc="Parsing RDF files", unit="file"):
        graph = Graph()

        try:
            # Parse this single file into a temporary graph.
            # We use publicID based on file:// URI so relative IRIs resolve safely.
            graph.parse(file_path.as_posix(), format=rdf_format, publicID=file_path.resolve().as_uri())
        except Exception as e:
            # For large messy dumps, it is often better to skip problematic files
            # than crash the entire extraction.
            tqdm.write(f"[WARN] Failed to parse {file_path}: {e}")
            continue

        # Inner progress bar: iterate triples from this graph
        triple_iter = graph.triples((None, None, None))
        triple_bar = tqdm(
            triple_iter,
            total=len(graph),
            desc=f"Triples in {file_path.name}",
            unit="triple",
            leave=False,
        )

        for s, p, o in triple_bar:
            # ------------------------------------------------------------------
            # 1) Register URI subjects as entities
            # ------------------------------------------------------------------
            if isinstance(s, URIRef):
                s_uri = str(s)
                if s_uri not in entities:
                    entities[s_uri] = {"label": None, "score": None}

            # ------------------------------------------------------------------
            # 2) Register URI objects as entities too
            #    This ensures "all entities" includes URI nodes that only appear
            #    as objects in triples.
            # ------------------------------------------------------------------
            if isinstance(o, URIRef):
                o_uri = str(o)
                if o_uri not in entities:
                    entities[o_uri] = {"label": None, "score": None}

            # ------------------------------------------------------------------
            # 3) If this triple gives a label to a URI subject, evaluate whether
            #    it is better than the label we already have.
            # ------------------------------------------------------------------
            if isinstance(s, URIRef) and isinstance(o, Literal):
                p_uri = str(p)

                if p_uri in LABEL_PREDICATE_PRIORITY:
                    candidate_text = clean_whitespace(str(o))
                    if candidate_text:
                        new_score = candidate_label_score(p_uri, o)

                        current_score = entities[str(s)]["score"]
                        if current_score is None or new_score > current_score:
                            entities[str(s)]["label"] = candidate_text
                            entities[str(s)]["score"] = new_score

        # Explicitly release the graph before moving to the next file.
        graph.close()
        del graph

    # Fill missing labels with URI-derived fallback labels
    for entity_uri, payload in entities.items():
        if payload["label"] is None:
            payload["label"] = uri_to_fallback_label(entity_uri)

    return entities


def build_output_json(entities: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """
    Convert the internal entity dictionary into the required SPARQL-results-like
    JSON structure.

    Output bindings are sorted by entity URI so the output is deterministic and
    matches `"ordered": true`.
    """
    bindings = []

    for entity_uri in sorted(entities.keys()):
        label = str(entities[entity_uri]["label"])

        bindings.append(
            {
                "id": {
                    "type": "uri",
                    "value": entity_uri,
                },
                "main_label": {
                    "type": "literal",
                    "value": label,
                },
            }
        )

    return {
        "head": {
            "link": [],
            "vars": ["id", "main_label"],
        },
        "results": {
            "distinct": False,
            "ordered": True,
            "bindings": bindings,
        },
    }


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all entities from a DOREMUS RDF dump into one JSON file."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the root of the DOREMUS dump directory.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path of the output JSON file.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON with indentation (larger file, easier to inspect).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_root}")

    print(f"Input dump:  {input_root}")
    print(f"Output JSON: {output_path}")

    entities = extract_entities(input_root)
    output = build_output_json(entities)

    # Make sure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )

    print(f"Done. Extracted {len(entities):,} entities.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()