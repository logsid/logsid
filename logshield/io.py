"""Read the frozen full-Content span data without changing text or gold."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path(os.environ.get("LOGSID_CACHE", Path.home() / ".cache" / "logsid-paper")).expanduser().resolve()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def taxonomy(data_dir):
    with (Path(data_dir) / "taxonomy.csv").open(encoding="utf-8-sig", newline="") as stream:
        return [row["category_id"] for row in csv.DictReader(stream)]


def source_inventory(data_dir):
    with (Path(data_dir) / "statistics.csv").open(encoding="utf-8-sig", newline="") as stream:
        return {row["Dataset"]: row for row in csv.DictReader(stream)}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(data_dir, sources):
    data_dir = Path(data_dir)
    paths = [data_dir / "taxonomy.csv", data_dir / "statistics.csv"]
    paths += [data_dir / (source + "_labels.jsonl.gz") for source in sources]
    return {path.name: file_sha256(path) for path in paths}


def iter_records(data_dir, source):
    """Yield every original row in native order, preserving Unicode offsets."""
    path = Path(data_dir) / (source + "_labels.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for line_id, line in enumerate(stream, 1):
            raw = json.loads(line)
            if raw["LineId"] != line_id or not isinstance(raw["Content"], str):
                raise ValueError(f"Invalid original row identity: {source}:{line_id}")
            yield {"source": source, "line_id": line_id, "timestamp": raw.get("Timestamp"),
                   "text": raw["Content"], "spans": raw["Labels"]}


def span_set(text, spans, categories):
    result = set()
    for span in spans:
        start, end, category = span["start"], span["end"], span["category"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text) or category not in categories):
            raise ValueError(f"Invalid Unicode [start,end) span: {span}")
        result.add((start, end, category))
    return result


ORDER_PROTOCOL = "chronological-history20-test80-relative-year-v2"


def build_order(data_dir, source, order_dir, progress=None):
    """Stably sort timestamps, unwrapping yearless Dec-to-Jan transitions.

    Yearless month/day values use a leap-year anchor and 366-day relative
    cycles solely for ordering. They do not acquire an observed year/timezone.
    Other backward timestamps stay in their cycle and are sorted normally.
    """
    import numpy as np
    size = int(source_inventory(data_dir)[source]["Logs"])
    if not 0 <= size <= np.iinfo(np.uint32).max:
        raise ValueError("Source size cannot be represented by the chronological index")
    stamps = np.zeros(size, dtype=np.int64)
    count = rollovers = 0
    yearless = None
    previous_month = None
    for row in iter_records(data_dir, source):
        count += 1
        if count > size:
            raise ValueError(f"Source exceeds its declared row count: {source}")
        value = row["timestamp"]
        current_yearless = isinstance(value, str) and value.startswith('--')
        if yearless is not None and current_yearless != yearless and isinstance(value, str):
            raise ValueError(f"Mixed full-date and yearless timestamps: {source}:{row['line_id']}")
        try:
            if not isinstance(value, str):
                raise ValueError('Timestamp must be a string')
            parsed = datetime.fromisoformat(
                ('2000-' + value[2:] if current_yearless else value).replace("Z", "+00:00"))
            if current_yearless and previous_month == 12 and parsed.month == 1:
                rollovers += 1
            previous_month = parsed.month
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            offset = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
            microseconds = ((offset.days * 86400 + offset.seconds) * 1000000 + offset.microseconds)
            if current_yearless:
                microseconds += rollovers * 366 * 86400 * 1000000
            stamps[row["line_id"] - 1] = microseconds
        except (TypeError, ValueError, AttributeError, OverflowError):
            raise ValueError(f"Missing or unparseable Timestamp at {source}:{row['line_id']}; "
                             "cannot construct chronological order") from None
        yearless = current_yearless
        if progress and count % 500000 == 0:
            progress({"source": source, "processed": count, "operation": "timestamp-index"})
    if count != size:
        raise ValueError(f"Incomplete source {source}: {count}/{size}")
    inversions = int(np.count_nonzero(stamps[1:] < stamps[:-1]))
    order = np.argsort(stamps, kind="stable").astype(np.uint32)
    mode = ("stable relative-year month/day/time order, ties by original LineId" if yearless
            else "stable Timestamp order, ties by original LineId")
    directory = Path(order_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (source + "_order.npy")
    np.save(path, order, allow_pickle=False)
    return {"rows": size, "history": size // 5, "test80": size - size // 5,
            "order_file": path.name, "order_sha256": file_sha256(path), "ordering": mode,
            "timestamp_naive_convention": "UTC for relative order only; no timezone inference",
            "yearless_timestamps": bool(yearless), "relative_year_rollovers": rollovers,
            "relative_year_convention": ("Leap-year 2000 month/day anchor, 366-day ordering cycles; "
                                         "advance only at native-order December-to-January transitions; "
                                         "not observed calendar years or elapsed time" if yearless else None),
            "missing_or_unparseable_timestamps": 0, "adjacent_timestamp_reversals": inversions}


def load_order(order_dir, source, *, expected_rows=None):
    """Validate order integrity and the first-20% / last-80% boundaries."""
    import numpy as np
    directory = Path(order_dir)
    meta = json.loads((directory / "metadata.json").read_text())
    if (meta.get("status") != "complete" or meta.get("protocol") != ORDER_PROTOCOL
            or source not in meta.get("sources", {})):
        raise ValueError("Chronological ordering is incomplete or incompatible")
    stats = meta["sources"][source]
    size = stats.get("rows")
    if (type(size) is not int or size < 0
            or (expected_rows is not None and size != expected_rows)
            or stats.get("history") != size // 5 or stats.get("test80") != size - size // 5
            or stats.get("order_file") != source + "_order.npy"):
        raise ValueError(f"Invalid chronological row counts or boundaries: {source}")
    path = directory / stats["order_file"]
    if file_sha256(path) != stats.get("order_sha256"):
        raise ValueError(f"Chronological order changed: {source}")
    order = np.load(path, mmap_mode="r", allow_pickle=False)
    if order.ndim != 1 or len(order) != size or order.dtype != np.uint32:
        raise ValueError(f"Malformed chronological order: {source}")
    if (size and int(order.max()) >= size) or np.any(np.bincount(order, minlength=size) != 1):
        raise ValueError(f"Chronological order must visit each original row exactly once: {source}")
    return order, meta
