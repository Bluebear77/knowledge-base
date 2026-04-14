#!/usr/bin/env python3
"""
Adaptive downloader for DOREMUS entities from https://data.doremus.org/sparql

Why this version is different
-----------------------------
The earlier version still timed out because each request asked the SPARQL server
to do a global GROUP BY + ORDER BY over a huge result set.

This version avoids that by:

1. Partitioning the crawl by the first characters of the entity label
   (for example: "a", "b", ..., "z", "0", ..., "9", and a fallback bucket).
2. Querying raw rows only:
      ?id ?main_label ?comment
   without GROUP_CONCAT or SAMPLE.
3. Merging duplicate rows client-side into the final combined JSON.
4. If one partition still times out, recursively splitting it:
      "a" -> "aa", "ab", "ac", ...
   until the partition becomes small enough.
5. Logging every failure in JSONL for later retry/debugging.

Important behavior
------------------
- One page JSON per successful request is stored in output_dir/pages/.
- A persistent checkpoint is saved after every successful page.
- A structured error log is written to errors.jsonl.
- The final combined JSON matches a SPARQL JSON result object with:
      head.vars = ["id", "main_label", "comment"]

Caveat
------
This is still a public endpoint. A very large crawl may still encounter
temporary 502/503/504 errors. This script is designed to survive those and
leave enough logs/checkpoints for resuming.
"""

from __future__ import annotations

import argparse
import json
import logging
import string
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PREFIXES = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"""

# Initial partition buckets.
# These are deliberately small and simple.
ROOT_BUCKETS = list(string.ascii_lowercase) + list(string.digits) + ["__OTHER__"]

# When a bucket times out, we split it into children with these suffixes.
SPLIT_CHARS = list(string.ascii_lowercase) + list(string.digits)


@dataclass
class ErrorRecord:
    timestamp: float
    bucket: str
    page_number: int
    last_id: str
    http_status: Optional[int]
    error_type: str
    error_message: str
    query: str


class DoremusAdaptiveFetcher:
    def __init__(
        self,
        endpoint: str,
        output_dir: Path,
        page_size: int = 100,
        request_timeout: int = 90,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        polite_delay: float = 0.5,
        max_bucket_depth: int = 3,
        user_agent: str = "doremus-adaptive-fetcher/2.0",
    ) -> None:
        self.endpoint = endpoint
        self.output_dir = output_dir
        self.page_size = page_size
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.polite_delay = polite_delay
        self.max_bucket_depth = max_bucket_depth
        self.user_agent = user_agent

        self.pages_dir = self.output_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.output_dir / "state.json"
        self.error_log_jsonl = self.output_dir / "errors.jsonl"
        self.text_log_path = self.output_dir / "run.log"
        self.combined_json_path = self.output_dir / "doremus_entities_combined.json"

        self.logger = self._setup_logger()
        self.session = self._build_session()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"DoremusAdaptiveFetcher[{id(self)}]")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(self.text_log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        return logger

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=0, connect=0, read=0, redirect=0, status=0)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Accept": "application/sparql-results+json",
                "User-Agent": self.user_agent,
            }
        )
        return session

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def load_state(self) -> Dict[str, Any]:
        """
        State structure:
        {
          "queue": [{"bucket":"a","depth":1}, ...],
          "done": ["a", "b", ...],
          "bucket_progress": {
             "a": {"last_id": "...", "page_number": 3}
          }
        }
        """
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))

        return {
            "queue": [{"bucket": b, "depth": 1} for b in ROOT_BUCKETS],
            "done": [],
            "bucket_progress": {},
        }

    def save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_error(
        self,
        bucket: str,
        page_number: int,
        last_id: str,
        exc: Exception,
        query: str,
    ) -> None:
        http_status = getattr(getattr(exc, "response", None), "status_code", None)
        record = ErrorRecord(
            timestamp=time.time(),
            bucket=bucket,
            page_number=page_number,
            last_id=last_id,
            http_status=http_status,
            error_type=type(exc).__name__,
            error_message=str(exc),
            query=query,
        )
        with self.error_log_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # SPARQL query building
    # ------------------------------------------------------------------

    def _bucket_filter(self, bucket: str) -> str:
        """
        Build the filter for one bucket.

        Bucket semantics:
        - "a"       => labels starting with "a"
        - "ab"      => labels starting with "ab"
        - "__OTHER__" => labels whose first character is not [a-z0-9]

        We lowercase labels to make partitioning stable.
        """
        label_expr = 'LCASE(STR(?main_label))'

        if bucket == "__OTHER__":
            return (
                f'FILTER(STRLEN({label_expr}) > 0)\n'
                f'  FILTER(!REGEX(SUBSTR({label_expr}, 1, 1), "^[a-z0-9]$"))'
            )

        bucket_json = json.dumps(bucket)
        return f'FILTER(STRSTARTS({label_expr}, {bucket_json}))'

    def build_query(self, bucket: str, last_id: Optional[str], limit: int) -> str:
        """
        Query raw rows only, leaving aggregation to Python.

        We keep ORDER BY simple and stable:
        ORDER BY STR(?id) STR(?main_label) STR(?comment)

        We page within a bucket using STR(?id) > "last_id".
        This is not perfect if the same id has many comment rows, but client-side
        merging makes it robust enough for practical crawling.
        """
        bucket_filter = self._bucket_filter(bucket)

        last_id_filter = ""
        if last_id:
            last_id_filter = f'\n  FILTER(STR(?id) > {json.dumps(last_id)})'

        query = f"""
{PREFIXES}
SELECT ?id ?main_label ?comment
WHERE {{
  ?id rdfs:label ?main_label .
  FILTER(isIRI(?id))
  {bucket_filter}
  OPTIONAL {{ ?id rdfs:comment ?comment }}{last_id_filter}
}}
ORDER BY STR(?id) STR(?main_label) STR(?comment)
LIMIT {int(limit)}
""".strip()

        return query

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------

    def fetch_page(self, bucket: str, page_number: int, last_id: Optional[str]) -> Dict[str, Any]:
        query = self.build_query(bucket=bucket, last_id=last_id, limit=self.page_size)

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(
                    "Requesting bucket=%s page=%s attempt=%s last_id=%r",
                    bucket,
                    page_number,
                    attempt,
                    last_id,
                )

                response = self.session.get(
                    self.endpoint,
                    params={
                        "query": query,
                        "format": "application/sparql-results+json",
                        # Some Virtuoso endpoints honor a timeout parameter.
                        # It does not hurt if ignored.
                        "timeout": str(self.request_timeout * 1000),
                    },
                    timeout=self.request_timeout,
                )

                if response.status_code != 200:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}: {response.text[:500]}",
                        response=response,
                    )

                payload = response.json()

                page_path = self.pages_dir / f"{bucket}__page_{page_number:06d}.json"
                page_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                return payload

            except Exception as exc:
                last_exc = exc
                self.log_error(bucket, page_number, last_id or "", exc, query)
                self.logger.warning(
                    "Failed bucket=%s page=%s attempt=%s error=%s",
                    bucket,
                    page_number,
                    attempt,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** (attempt - 1))

        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Bucket splitting
    # ------------------------------------------------------------------

    def split_bucket(self, bucket: str, depth: int) -> List[Dict[str, Any]]:
        """
        Recursively split a problematic bucket into finer buckets.

        Example:
        "a" -> ["aa", "ab", ..., "az", "a0", ..., "a9"]
        """
        if bucket == "__OTHER__":
            # This bucket is awkward to split generically by label prefix.
            # Keep it unsplit and let the caller decide whether to skip/fail.
            return []

        if depth >= self.max_bucket_depth:
            return []

        return [{"bucket": bucket + ch, "depth": depth + 1} for ch in SPLIT_CHARS]

    # ------------------------------------------------------------------
    # Combine pages
    # ------------------------------------------------------------------

    def combine_pages(self) -> Dict[str, Any]:
        """
        Merge all per-page raw JSON files into one SPARQL JSON object.

        Because we query raw rows, the same ?id can appear multiple times due to:
        - multiple labels
        - multiple comments
        - label/comment combinations

        We merge by id:
        - main_label: first non-empty label seen
        - comment: unique comments joined with ;;;
        """
        merged: Dict[str, Dict[str, Any]] = {}

        for page_file in sorted(self.pages_dir.glob("*.json")):
            payload = json.loads(page_file.read_text(encoding="utf-8"))
            bindings = payload.get("results", {}).get("bindings", [])

            for row in bindings:
                if "id" not in row:
                    continue

                entity_id = row["id"]["value"]
                label = row.get("main_label", {}).get("value")
                comment = row.get("comment", {}).get("value")

                item = merged.setdefault(
                    entity_id,
                    {
                        "id": {
                            "type": "uri",
                            "value": entity_id,
                        },
                        "main_label": None,
                        "_comments": set(),
                    },
                )

                if label and item["main_label"] is None:
                    item["main_label"] = {
                        "type": "literal",
                        "value": label,
                    }

                if comment:
                    item["_comments"].add(comment)

        final_bindings: List[Dict[str, Any]] = []

        for entity_id in sorted(merged.keys()):
            item = merged[entity_id]
            out_row: Dict[str, Any] = {
                "id": item["id"],
            }

            if item["main_label"] is not None:
                out_row["main_label"] = item["main_label"]

            comments = sorted(item["_comments"])
            if comments:
                out_row["comment"] = {
                    "type": "literal",
                    "value": ";;;".join(comments),
                }

            final_bindings.append(out_row)

        combined = {
            "head": {
                "link": [],
                "vars": ["id", "main_label", "comment"],
            },
            "results": {
                "distinct": False,
                "ordered": True,
                "bindings": final_bindings,
            },
        }

        self.combined_json_path.write_text(
            json.dumps(combined, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return combined

    # ------------------------------------------------------------------
    # Main crawl
    # ------------------------------------------------------------------

    def run(self, max_buckets: Optional[int] = None) -> Dict[str, Any]:
        state = self.load_state()
        queue: List[Dict[str, Any]] = state["queue"]
        done: List[str] = state["done"]
        bucket_progress: Dict[str, Dict[str, Any]] = state["bucket_progress"]

        processed_buckets = 0

        self.logger.info(
            "Starting adaptive crawl. queued_buckets=%s done_buckets=%s",
            len(queue),
            len(done),
        )

        while queue:
            if max_buckets is not None and processed_buckets >= max_buckets:
                self.logger.info("Stopping because max_buckets=%s was reached", max_buckets)
                break

            bucket_info = queue.pop(0)
            bucket = bucket_info["bucket"]
            depth = int(bucket_info["depth"])

            progress = bucket_progress.get(bucket, {})
            last_id = progress.get("last_id")
            page_number = int(progress.get("page_number", 0))

            self.logger.info(
                "Working on bucket=%s depth=%s resume_page=%s resume_last_id=%r",
                bucket,
                depth,
                page_number,
                last_id,
            )

            try:
                while True:
                    next_page = page_number + 1
                    payload = self.fetch_page(bucket=bucket, page_number=next_page, last_id=last_id)
                    bindings = payload.get("results", {}).get("bindings", [])

                    if not bindings:
                        self.logger.info("Bucket=%s finished (empty page)", bucket)
                        done.append(bucket)
                        bucket_progress.pop(bucket, None)
                        break

                    last_id = bindings[-1]["id"]["value"]
                    page_number = next_page
                    bucket_progress[bucket] = {
                        "last_id": last_id,
                        "page_number": page_number,
                    }
                    self.save_state(
                        {
                            "queue": queue,
                            "done": done,
                            "bucket_progress": bucket_progress,
                        }
                    )

                    self.logger.info(
                        "Saved bucket=%s page=%s rows=%s next_last_id=%r",
                        bucket,
                        page_number,
                        len(bindings),
                        last_id,
                    )

                    time.sleep(self.polite_delay)

                    if len(bindings) < self.page_size:
                        self.logger.info("Bucket=%s finished (last short page)", bucket)
                        done.append(bucket)
                        bucket_progress.pop(bucket, None)
                        break

            except Exception as exc:
                self.logger.warning(
                    "Bucket=%s failed at depth=%s and will be split if possible. error=%s",
                    bucket,
                    depth,
                    exc,
                )

                children = self.split_bucket(bucket, depth=depth)
                if children:
                    self.logger.info(
                        "Splitting bucket=%s into %s children",
                        bucket,
                        len(children),
                    )
                    # Put children at the front so the crawler immediately continues there.
                    queue = children + queue
                    # Discard partial progress for the parent bucket.
                    bucket_progress.pop(bucket, None)
                else:
                    self.logger.error(
                        "Bucket=%s cannot be split further; leaving it unfinished. "
                        "Check errors.jsonl and consider lowering page-size or increasing max-bucket-depth.",
                        bucket,
                    )

                self.save_state(
                    {
                        "queue": queue,
                        "done": done,
                        "bucket_progress": bucket_progress,
                    }
                )

            processed_buckets += 1

        combined = self.combine_pages()
        self.logger.info(
            "Combined %s merged entities into %s",
            len(combined["results"]["bindings"]),
            self.combined_json_path,
        )
        return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive DOREMUS entity crawler for the public SPARQL endpoint."
    )
    parser.add_argument(
        "--endpoint",
        default="https://data.doremus.org/sparql",
        help="SPARQL endpoint URL",
    )
    parser.add_argument(
        "--output-dir",
        default="doremus_entities_adaptive",
        help="Output directory for pages, logs, state, and final JSON",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Rows per request. Start small on a public endpoint.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=90,
        help="HTTP timeout in seconds for one request",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retries for one page before splitting the bucket",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Exponential backoff base",
    )
    parser.add_argument(
        "--polite-delay",
        type=float,
        default=0.5,
        help="Delay between successful requests",
    )
    parser.add_argument(
        "--max-bucket-depth",
        type=int,
        default=3,
        help="Maximum recursive bucket depth. Example: a -> aa -> aaa",
    )
    parser.add_argument(
        "--max-buckets",
        type=int,
        default=None,
        help="Optional testing stop after N bucket attempts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fetcher = DoremusAdaptiveFetcher(
        endpoint=args.endpoint,
        output_dir=Path(args.output_dir),
        page_size=args.page_size,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        polite_delay=args.polite_delay,
        max_bucket_depth=args.max_bucket_depth,
    )
    fetcher.run(max_buckets=args.max_buckets)


if __name__ == "__main__":
    main()
