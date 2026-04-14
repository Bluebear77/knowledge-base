#!/usr/bin/env python3
"""
Doremus dump downloader and extractor.

What this script does:
1. Starts from a base directory listing URL.
2. Recursively discovers all files reachable under that URL.
3. Estimates file sizes when possible.
4. Downloads each file into a matching local directory structure.
5. Skips files that already exist locally with the expected size.
6. Finds any downloaded `.tar.gz` archives and optionally extracts them.

Typical usage:
    python doremus_downloader.py
    python doremus_downloader.py https://data.doremus.org/dump/ my_output_dir

Command-line arguments:
    argv[1] -> optional base URL to scan
    argv[2] -> optional destination directory

Important implementation details:
- HTML directory listings are parsed by collecting <a href="..."> links.
- Only links that remain under the original base path are followed.
- Archive extraction includes a path-safety check to prevent files from
  escaping the intended extraction directory.
- File size detection first tries HTTP HEAD, then falls back to GET if needed.

Limitations:
- This assumes the remote server exposes directory listings via HTML links.
- It does not resume partial downloads.
- It does not verify checksums.
- If the server does not provide Content-Length, size may be reported as unknown.
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Set

# Default remote directory to scan for downloadable files.
BASE_URL = "https://data.doremus.org/dump/"

# Default local directory where downloaded files will be stored.
DEST_DIR = Path("doremus_dump")

# Timeout in seconds for HTTP requests.
TIMEOUT = 60

# Number of bytes to read per iteration while downloading.
# 1 MiB is a reasonable chunk size for large file streaming.
CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class RemoteFile:
    """
    Represents one discovered file on the remote server.

    Fields:
    - url: full remote URL of the file
    - rel_path: relative path under the base URL, used to reproduce the
      remote directory structure locally
    - size: file size in bytes if known, otherwise None
    """
    url: str
    rel_path: PurePosixPath
    size: int | None


class LinkParser(HTMLParser):
    """
    Minimal HTML parser that collects all hyperlink targets (<a href="...">).

    This is used to parse simple directory listing pages exposed by the server.
    """

    def __init__(self) -> None:
        super().__init__()
        # Store every href encountered while parsing.
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        """
        Called by HTMLParser for every opening tag.

        We only care about <a> tags, because those contain links to files
        and subdirectories in the remote directory listing.
        """
        if tag.lower() != "a":
            return

        # Convert the list of (key, value) pairs into a dict so we can
        # easily retrieve the href attribute.
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def format_size(num_bytes: int | None) -> str:
    """
    Convert a byte count into a human-readable size string.

    Examples:
    - 512 -> "512.00 B"
    - 2048 -> "2.00 KiB"
    - None -> "unknown"
    """
    if num_bytes is None:
        return "unknown"

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)

    # Repeatedly divide by 1024 until we find the most suitable unit.
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    # Fallback, though normal flow should always return inside the loop.
    return f"{num_bytes} B"


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """
    Repeatedly ask the user a yes/no question until a valid answer is given.

    Parameters:
    - prompt: the question text shown to the user
    - default: the answer used if the user presses Enter without typing anything

    Returns:
    - True for yes
    - False for no
    """
    # Display the default choice in the prompt:
    # [Y/n] means default yes, [y/N] means default no.
    suffix = "[Y/n]" if default else "[y/N]"

    while True:
        answer = input(f"{prompt} {suffix} ").strip().lower()

        # Empty input means "use the default".
        if not answer:
            return default

        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False

        print("Please answer yes or no.")


def http_request(url: str, method: str = "GET"):
    """
    Build and execute an HTTP request with a custom User-Agent and timeout.

    This centralizes request creation so all network calls behave consistently.
    """
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "python-urllib/doremus-downloader"},
    )
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def get_links(url: str) -> list[str]:
    """
    Download an HTML page and extract all href values from its links.

    Intended for parsing directory listing pages on the remote server.
    """
    with http_request(url, method="GET") as resp:
        # Use the charset declared by the server if available,
        # otherwise assume UTF-8.
        charset = resp.headers.get_content_charset() or "utf-8"
        html = resp.read().decode(charset, errors="replace")

    parser = LinkParser()
    parser.feed(html)
    return parser.hrefs


def head_size(url: str) -> int | None:
    """
    Try to determine the size of a remote file.

    Strategy:
    1. Attempt an HTTP HEAD request and read Content-Length.
    2. If that fails, fall back to GET:
       - use Content-Length if present
       - otherwise read the response body and measure its length

    Returns:
    - size in bytes if determinable
    - None if the size cannot be determined
    """
    try:
        with http_request(url, method="HEAD") as resp:
            length = resp.headers.get("Content-Length")
            if length is not None:
                return int(length)
    except Exception:
        # Some servers do not support HEAD properly, so failure here is okay.
        pass

    try:
        with http_request(url, method="GET") as resp:
            length = resp.headers.get("Content-Length")
            if length is not None:
                return int(length)

            # Last resort: download the entire response into memory and
            # measure it. This is not ideal for large files, but it allows
            # us to estimate size when Content-Length is missing.
            data = resp.read()
            return len(data)
    except Exception:
        return None


def split_parts(path: str) -> tuple[str, ...]:
    """
    Split a POSIX-style path into clean path components.

    Filters out root markers and empty segments so comparisons are easier.
    """
    return tuple(part for part in PurePosixPath(path).parts if part not in {"/", ""})


def compute_rel_path(base_url: str, file_url: str) -> PurePosixPath:
    """
    Compute the file path relative to the base URL.

    Example:
    - base_url: https://host/data/
    - file_url: https://host/data/sub/file.txt
    - result:   sub/file.txt

    Raises:
    - ValueError if the file is not actually under the base path
    - ValueError if no relative path can be derived
    """
    base_parts = split_parts(urllib.parse.urlparse(base_url).path)
    file_parts = split_parts(urllib.parse.urlparse(file_url).path)

    # Ensure the file really lies under the scanned base path.
    if file_parts[: len(base_parts)] != base_parts:
        raise ValueError(f"Unexpected file outside base path: {file_url}")

    # Remove the base path prefix to get the local relative path.
    rel_parts = file_parts[len(base_parts) :]
    if not rel_parts:
        raise ValueError(f"Could not derive relative path for {file_url}")

    return PurePosixPath(*rel_parts)


def discover_files(base_url: str) -> list[RemoteFile]:
    """
    Recursively crawl the base URL and discover all reachable files.

    This function performs a breadth-first traversal over directory links:
    - queue holds directories still to visit
    - visited_dirs prevents revisiting the same directory
    - seen_files prevents duplicate file entries

    Returns:
    - a sorted list of RemoteFile objects
    """
    queue: list[str] = [base_url]
    visited_dirs: Set[str] = set()
    seen_files: Set[str] = set()
    files: list[RemoteFile] = []

    while queue:
        # pop(0) makes this behave like a FIFO queue (breadth-first traversal).
        current = queue.pop(0)

        if current in visited_dirs:
            continue
        visited_dirs.add(current)

        # Extract all href links from the current directory listing page.
        for href in get_links(current):
            # Resolve relative links against the current page URL.
            absolute = urllib.parse.urljoin(current, href)
            parsed = urllib.parse.urlparse(absolute)

            # Ignore non-web links such as mailto:, javascript:, etc.
            if parsed.scheme not in {"http", "https"}:
                continue

            # Ignore common self/parent directory entries.
            if href in {"../", "./"}:
                continue

            # Prevent crawling outside the original base directory tree.
            base_path = urllib.parse.urlparse(base_url).path
            if not parsed.path.startswith(base_path):
                continue

            # Some directory listings expose "index.html" links that represent
            # the current page rather than an actual downloadable file.
            name = PurePosixPath(parsed.path).name
            if name.startswith("index.html"):
                continue

            # Convention: href ending with "/" is treated as a subdirectory.
            if href.endswith("/"):
                if absolute not in visited_dirs:
                    queue.append(absolute)
            else:
                # Skip duplicate files if already seen.
                if absolute in seen_files:
                    continue
                seen_files.add(absolute)

                files.append(
                    RemoteFile(
                        url=absolute,
                        rel_path=compute_rel_path(base_url, absolute),
                        size=head_size(absolute),
                    )
                )

    # Sort results for predictable output and download order.
    files.sort(key=lambda f: f.rel_path.as_posix())
    return files


def download_file(remote: RemoteFile, dest_root: Path) -> Path:
    """
    Download one remote file into the local destination tree.

    Behavior:
    - Creates parent directories as needed
    - Skips download if a local file already exists with the expected size
    - Streams file content in chunks to avoid loading the whole file in memory

    Returns:
    - the local destination path
    """
    dest_path = dest_root / remote.rel_path

    # Ensure the destination directory exists before writing the file.
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # If the file already exists locally, compare sizes.
    # A matching known size means we can safely skip re-downloading.
    existing_size = dest_path.stat().st_size if dest_path.exists() else -1
    if remote.size is not None and existing_size == remote.size:
        print(f"[skip] {remote.rel_path} ({format_size(remote.size)})")
        return dest_path

    print(f"[get ] {remote.rel_path} ({format_size(remote.size)})")

    # Stream the response body in chunks to avoid high memory usage.
    with http_request(remote.url, method="GET") as resp, open(dest_path, "wb") as fh:
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            fh.write(chunk)

    return dest_path


def safe_extract_tar_gz(archive_path: Path, extract_dir: Path) -> None:
    """
    Safely extract a .tar.gz archive into the target directory.

    Security measure:
    Before extraction, every member path is resolved and checked to ensure it
    stays inside the intended extraction directory. This protects against
    path traversal entries such as "../../../etc/passwd".

    Uses tar.extractall(filter="data") when supported by the Python version.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:gz") as tar:
        base_dir = extract_dir.resolve()

        for member in tar.getmembers():
            # Compute the final resolved target path for each archive member.
            target = (extract_dir / member.name).resolve()

            # Reject the archive if any member would escape the extraction root.
            if os.path.commonpath([str(base_dir), str(target)]) != str(base_dir):
                raise RuntimeError(f"Unsafe path in archive {archive_path}: {member.name}")

        try:
            # Python 3.12+ supports filter="data", which blocks some dangerous
            # tar features and is safer for untrusted archives.
            tar.extractall(path=extract_dir, filter="data")
        except TypeError:
            # Fallback for older Python versions that do not support "filter".
            tar.extractall(path=extract_dir)


def extract_all_archives(dest_root: Path) -> None:
    """
    Locate all downloaded .tar.gz files under dest_root and optionally extract them.

    Each archive is extracted into a sibling directory with the same base name:
    - foo.tar.gz -> foo/
    """
    archives = sorted(dest_root.rglob("*.tar.gz"))
    if not archives:
        print("No .tar.gz archives found.")
        return

    total_archives_size = sum(p.stat().st_size for p in archives)
    print(f"Found {len(archives)} .tar.gz archives totalling {format_size(total_archives_size)} compressed.")

    # Ask the user before performing extraction, since this may consume
    # significant disk space and time.
    if not ask_yes_no("Should I extract ALL now?", default=False):
        print("Extraction cancelled.")
        return

    for archive in archives:
        # Remove the ".tar.gz" suffix to produce the extraction directory name.
        extract_dir = archive.parent / archive.name[:-7]
        print(f"[xtr ] {archive.relative_to(dest_root)} -> {extract_dir.relative_to(dest_root)}")
        safe_extract_tar_gz(archive, extract_dir)

    print("Extraction finished.")


def main() -> int:
    """
    Program entry point.

    Flow:
    1. Read optional CLI arguments
    2. Normalize base URL
    3. Discover all remote files
    4. Print summary statistics
    5. Download all files
    6. Offer to extract all .tar.gz archives
    """
    base_url = BASE_URL
    dest_dir = DEST_DIR

    # Allow the user to override defaults from the command line.
    if len(sys.argv) >= 2:
        base_url = sys.argv[1]
    if len(sys.argv) >= 3:
        dest_dir = Path(sys.argv[2])

    # Normalize the base URL so directory joining behaves consistently.
    if not base_url.endswith("/"):
        base_url += "/"

    print(f"Scanning {base_url} ...")
    files = discover_files(base_url)

    if not files:
        print("No files found.")
        return 1

    # Compute some summary stats before downloading.
    total_size = sum(f.size or 0 for f in files)
    unknown_count = sum(1 for f in files if f.size is None)
    archive_count = sum(1 for f in files if f.rel_path.name.endswith(".tar.gz"))

    print(f"Found {len(files)} files under {base_url}")
    print(f"Estimated total download size before extraction: {format_size(total_size)}")
    print(f"Found {archive_count} .tar.gz archives")

    # If some file sizes are unknown, total_size is only a lower bound.
    if unknown_count:
        print(f"Warning: {unknown_count} files had unknown size, so the real total may be larger.")

    # Ensure the destination root exists.
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download each discovered file.
    for remote in files:
        download_file(remote, dest_dir)

    print("Download finished.")
    extract_all_archives(dest_dir)
    return 0


if __name__ == "__main__":
    # Raise SystemExit so the return code from main() becomes the process exit code.
    raise SystemExit(main())