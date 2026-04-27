#!/usr/bin/env python3

# ============================================================
# HOW TO RUN
# ============================================================
# 1. Install dependencies:
#       pip install requests tqdm
#
# 2. Edit the hardcoded file paths in the CONFIG section below.
#
# 3. Run:
#       python wikidata_local_property_csv_hardcoded.py
#
# ============================================================
# WHAT THIS SCRIPT DOES
# ============================================================
# Input CSV requirements:
#   The input CSV must contain these two columns:
#       Entity_1
#       Entity_2
#
# The values in these columns may be either:
#   - raw QIDs, such as:
#         Q567
#         Q1055
#
#   - or human-readable text containing a QID, such as:
#         Angela Merkel (Q567)
#         Hamburg (Q1055)
#
# Output files:
#
# 1) output.csv
#    - keeps the original Entity_1 and Entity_2 values from the input
#    - adds:
#         Property_Number
#         Propery_Name
#    - Propery_Name contains raw property IDs, for example:
#         P19;P27
#
# 2) output_label_name.csv
#    - converts Entity_1 and Entity_2 to readable labels:
#         Angela Merkel (Q567)
#         Hamburg (Q1055)
#    - converts Propery_Name to readable property labels:
#         place of birth (P19);country of citizenship (P27)
#
# IMPORTANT INTERPRETATION:
# This script does NOT search for arbitrary graph paths and return all
# properties on those paths.
#
# Instead, it returns the DISTINCT first-hop properties on Entity_1
# whose value can reach Entity_2 within MAX_HOPS.
#
# Example:
#   Q567 --P19--> Eimsbüttel --P131--> Q1055
#
# Then the script returns:
#   Property_Number = 1
#   Propery_Name    = P19
#
# If more than one first-hop property from Entity_1 can eventually reach
# Entity_2 within the hop limit, all such first-hop properties are returned.
# ============================================================

from __future__ import annotations

import csv
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
# Change these paths to your own files.
INPUT_CSV_PATH = "input.csv"
OUTPUT_CSV_PATH = "output.csv"
OUTPUT_LABEL_CSV_PATH = "output_label_name.csv"

# Maximum number of outward hops allowed from Entity_1 to Entity_2.
# Example:
#   MAX_HOPS = 1  means only direct connection:
#       Entity_1 --Pxx--> Entity_2
#
#   MAX_HOPS = 2  means:
#       Entity_1 --Pxx--> intermediate --Pyy--> Entity_2
MAX_HOPS = 2

# Label language used in output_label_name.csv
LABEL_LANGUAGE = "en"

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
WBGETENTITIES_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "WikidataLocalPropertyCSVChecker/1.0 (Python requests; contact: your-email@example.com)",
}


# ============================================================
# INPUT PARSING / VALIDATION
# ============================================================

def extract_qid_from_cell(cell_value: str) -> str:
    """
    Extract a QID from a CSV cell.

    This function supports both of these formats:
        Q567
        Angela Merkel (Q567)

    It finds the first substring that looks like a Wikidata QID.

    Parameters
    ----------
    cell_value : str
        Raw CSV cell content.

    Returns
    -------
    str
        Normalized uppercase QID, such as "Q567".

    Raises
    ------
    ValueError
        If no valid QID can be found in the input text.
    """
    text = str(cell_value).strip().upper()

    # Search for the first QID-looking token anywhere in the string.
    match = re.search(r"Q[1-9]\d*", text)
    if not match:
        raise ValueError(f"Could not find a valid QID in: {cell_value}")

    return match.group(0)


def extract_pid_list_from_joined_string(value: str) -> List[str]:
    """
    Split a joined property string like:
        P19;P27;P131

    into:
        ["P19", "P27", "P131"]

    Empty strings return an empty list.
    """
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def validate_qid(qid: str) -> str:
    """
    Validate a Wikidata item ID (QID).
    """
    qid = str(qid).strip().upper()
    if not re.fullmatch(r"Q[1-9]\d*", qid):
        raise ValueError(f"Invalid QID: {qid}")
    return qid


def validate_pid(pid: str) -> str:
    """
    Validate a Wikidata property ID (PID).
    """
    pid = str(pid).strip().upper()
    if not re.fullmatch(r"P[1-9]\d*", pid):
        raise ValueError(f"Invalid PID: {pid}")
    return pid


# ============================================================
# NETWORK HELPERS
# ============================================================

def entity_exists(qid: str, timeout: int = 20) -> bool:
    """
    Check whether a Wikidata entity exists.

    We call Special:EntityData/QID.json.
    If the HTTP status is 200, the entity exists.
    """
    url = ENTITY_DATA_URL.format(qid)
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": HEADERS["User-Agent"]},
    )
    return response.status_code == 200


def run_sparql(query: str, timeout: int = 30, retries: int = 2) -> Dict[str, Any]:
    """
    Execute a SPARQL query against Wikidata and return JSON.

    Retries are included because the public Wikidata endpoint may
    occasionally fail temporarily.
    """
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            response = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"SPARQL query failed: {exc}") from exc

    raise RuntimeError(f"SPARQL query failed: {last_error}")


# ============================================================
# LABEL HELPERS
# ============================================================

def get_entity_or_property_label(entity_id: str, language: str = LABEL_LANGUAGE) -> str:
    """
    Fetch a human-readable label for a Wikidata QID or PID.

    Examples:
        Q567  -> Angela Merkel
        Q1055 -> Hamburg
        P19   -> place of birth

    If a label cannot be retrieved, the function returns the ID itself.

    Parameters
    ----------
    entity_id : str
        QID or PID.
    language : str
        Requested label language.

    Returns
    -------
    str
        Human-readable label if available, else the raw ID.
    """
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels",
        "languages": language,
        "format": "json",
    }

    try:
        response = requests.get(
            WBGETENTITIES_API,
            params=params,
            headers={"User-Agent": HEADERS["User-Agent"]},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        entity_data = data.get("entities", {}).get(entity_id, {})
        labels = entity_data.get("labels", {})
        if language in labels:
            return labels[language]["value"]

    except Exception:
        pass

    return entity_id


def build_readable_entity_cell(original_value: str, language: str = LABEL_LANGUAGE) -> str:
    """
    Convert an entity cell into a human-readable format:
        Angela Merkel (Q567)
        Hamburg (Q1055)

    If the QID cannot be extracted, return the original value unchanged.
    """
    try:
        qid = extract_qid_from_cell(original_value)
        label = get_entity_or_property_label(qid, language=language)
        return f"{label} ({qid})"
    except Exception:
        return str(original_value)


def build_readable_property_cell(property_value: str, language: str = LABEL_LANGUAGE) -> str:
    """
    Convert a semicolon-separated PID string like:
        P19;P27

    into:
        place of birth (P19);country of citizenship (P27)

    Empty values stay empty.
    """
    pids = extract_pid_list_from_joined_string(property_value)
    if not pids:
        return ""

    readable_parts: List[str] = []
    for pid in pids:
        label = get_entity_or_property_label(pid, language=language)
        readable_parts.append(f"{label} ({pid})")

    return ";".join(readable_parts)


# ============================================================
# SPARQL QUERY CONSTRUCTION
# ============================================================

def extract_last_path_segment(uri: str) -> str:
    """
    Extract the last segment from a Wikidata URI.

    Example:
        http://www.wikidata.org/prop/direct/P19 -> P19
    """
    return uri.rsplit("/", 1)[-1]


def build_first_hop_property_query(qid1: str, qid2: str, max_hops: int) -> str:
    """
    Build a SPARQL query that returns DISTINCT first-hop properties from Entity_1
    that can reach Entity_2 within max_hops.

    IMPORTANT:
    - The FIRST edge is always outgoing from Entity_1.
    - Therefore, the returned property is guaranteed to be a property of Entity_1.
    - Only direct Wikidata truthy entity-to-entity edges are traversed.

    Example with max_hops = 2:
        wd:Q567 ?p1 ?n1 .
        ?n1 ?p2 wd:Q1055 .

    If this matches, the query returns ?p1, not ?p2.

    This is exactly what we want when we say:
        "Which property on Entity_1 connects it to Entity_2
         through the local statement graph?"
    """
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")

    union_blocks: List[str] = []

    # Case 1: direct one-hop connection
    union_blocks.append(f"""
    {{
      wd:{qid1} ?p1 wd:{qid2} .
      FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
    }}
    """)

    # Case 2..N: outward multi-hop paths
    if max_hops >= 2:
        for hops in range(2, max_hops + 1):
            lines = []
            lines.append("{")
            lines.append(f"  wd:{qid1} ?p1 ?n1 .")
            lines.append('  FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))')
            lines.append('  FILTER(STRSTARTS(STR(?n1), "http://www.wikidata.org/entity/Q"))')

            # Intermediate outward edges
            for i in range(1, hops - 1):
                lines.append(f"  ?n{i} ?p{i+1} ?n{i+1} .")
                lines.append(
                    f'  FILTER(STRSTARTS(STR(?p{i+1}), "http://www.wikidata.org/prop/direct/"))'
                )
                lines.append(
                    f'  FILTER(STRSTARTS(STR(?n{i+1}), "http://www.wikidata.org/entity/Q"))'
                )

            # Final edge reaches Entity_2
            lines.append(f"  ?n{hops-1} ?p{hops} wd:{qid2} .")
            lines.append(
                f'  FILTER(STRSTARTS(STR(?p{hops}), "http://www.wikidata.org/prop/direct/"))'
            )

            # Mild anti-cycle constraints on intermediate nodes
            for i in range(1, hops):
                for j in range(i + 1, hops):
                    lines.append(f"  FILTER(?n{i} != ?n{j})")

            lines.append("}")
            union_blocks.append("\n".join(lines))

    query = "SELECT DISTINCT ?p1 WHERE {\n"
    query += "\nUNION\n".join(union_blocks)
    query += "\n}\nORDER BY ?p1"

    return query


def find_connecting_first_hop_properties(qid1: str, qid2: str, max_hops: int) -> List[str]:
    """
    Return all DISTINCT first-hop property IDs on Entity_1 that can reach Entity_2
    within max_hops.

    Example:
        Q567 --P19--> Eimsbüttel --P131--> Q1055

    returns:
        ["P19"]

    If multiple first-hop properties work, they are all returned.
    """
    query = build_first_hop_property_query(qid1, qid2, max_hops)
    data = run_sparql(query)
    bindings = data.get("results", {}).get("bindings", [])

    property_ids: List[str] = []
    seen: Set[str] = set()

    for row in bindings:
        prop_uri = row["p1"]["value"]
        pid = extract_last_path_segment(prop_uri)
        if pid not in seen:
            seen.add(pid)
            property_ids.append(pid)

    return property_ids


def summarize_properties(property_ids: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Convert a list of property IDs into:
        Property_Number
        Propery_Name

    If there are no properties, return (None, None),
    so the output CSV cells stay empty.
    """
    if not property_ids:
        return None, None
    return len(property_ids), ";".join(property_ids)


# ============================================================
# CSV PROCESSING
# ============================================================

def read_input_rows(input_csv_path: str) -> List[Dict[str, str]]:
    """
    Read the input CSV into memory.

    We load all rows first so that:
    - tqdm can show total progress
    - we can write two output files afterward

    Required columns:
        Entity_1
        Entity_2
    """
    with open(input_csv_path, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)

        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        required = {"Entity_1", "Entity_2"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        return list(reader)


def process_rows(rows: List[Dict[str, str]], max_hops: int) -> List[Dict[str, str]]:
    """
    Process each input row and produce enriched output rows.

    The returned rows keep all original columns and add:
        Property_Number
        Propery_Name

    Propery_Name in this stage contains raw PIDs, not readable labels.
    """
    processed_rows: List[Dict[str, str]] = []

    # Cache entity existence checks to avoid repeating the same network call.
    existence_cache: Dict[str, bool] = {}

    for row in tqdm(rows, desc="Processing rows", unit="row"):
        raw_entity_1 = row.get("Entity_1", "")
        raw_entity_2 = row.get("Entity_2", "")

        property_number: Optional[int] = None
        propery_name: Optional[str] = None

        try:
            qid1 = validate_qid(extract_qid_from_cell(raw_entity_1))
            qid2 = validate_qid(extract_qid_from_cell(raw_entity_2))

            if qid1 not in existence_cache:
                existence_cache[qid1] = entity_exists(qid1)

            if qid2 not in existence_cache:
                existence_cache[qid2] = entity_exists(qid2)

            if existence_cache[qid1] and existence_cache[qid2]:
                property_ids = find_connecting_first_hop_properties(
                    qid1=qid1,
                    qid2=qid2,
                    max_hops=max_hops,
                )
                property_number, propery_name = summarize_properties(property_ids)

        except Exception as exc:
            print(
                f"Warning: could not process row with values "
                f"({raw_entity_1}, {raw_entity_2}): {exc}",
                file=sys.stderr,
            )

        new_row = dict(row)
        new_row["Property_Number"] = "" if property_number is None else property_number
        new_row["Propery_Name"] = "" if propery_name is None else propery_name
        processed_rows.append(new_row)

    return processed_rows


def write_output_csv(output_csv_path: str, rows: List[Dict[str, str]]) -> None:
    """
    Write the raw-ID output CSV.

    This file preserves the original Entity_1 and Entity_2 values from the input,
    and writes raw property IDs in Propery_Name.
    """
    if not rows:
        # If input is empty, still write a valid header.
        fieldnames = ["Entity_1", "Entity_2", "Property_Number", "Propery_Name"]
    else:
        fieldnames = list(rows[0].keys())

    with open(output_csv_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_output_label_csv(output_label_csv_path: str, rows: List[Dict[str, str]], language: str) -> None:
    """
    Write the human-readable output_label_name.csv.

    Differences from output.csv:
    - Entity_1 becomes, for example:
          Angela Merkel (Q567)
    - Entity_2 becomes, for example:
          Hamburg (Q1055)
    - Propery_Name becomes, for example:
          place of birth (P19);country of citizenship (P27)
    """
    if not rows:
        fieldnames = ["Entity_1", "Entity_2", "Property_Number", "Propery_Name"]
    else:
        fieldnames = list(rows[0].keys())

    readable_rows: List[Dict[str, str]] = []

    for row in tqdm(rows, desc="Writing readable labels", unit="row"):
        readable_row = dict(row)

        readable_row["Entity_1"] = build_readable_entity_cell(row.get("Entity_1", ""), language=language)
        readable_row["Entity_2"] = build_readable_entity_cell(row.get("Entity_2", ""), language=language)
        readable_row["Propery_Name"] = build_readable_property_cell(row.get("Propery_Name", ""), language=language)

        readable_rows.append(readable_row)

    with open(output_label_csv_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(readable_rows)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """
    Main execution function.

    This version uses only hardcoded paths from the CONFIG section.
    No command-line arguments are required.
    """
    try:
        rows = read_input_rows(INPUT_CSV_PATH)
        processed_rows = process_rows(rows, max_hops=MAX_HOPS)
        write_output_csv(OUTPUT_CSV_PATH, processed_rows)
        write_output_label_csv(OUTPUT_LABEL_CSV_PATH, processed_rows, language=LABEL_LANGUAGE)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Done.")
    print(f"Raw output written to: {OUTPUT_CSV_PATH}")
    print(f"Readable-label output written to: {OUTPUT_LABEL_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())