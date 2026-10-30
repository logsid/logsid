"""Disk-backed original logs; this storage layer defines no training split."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sqlite3
from pathlib import Path

from .io import DEFAULT_CACHE, ROOT, atomic_json, file_sha256, source_inventory, span_set, taxonomy

DEFAULT_INDEX = DEFAULT_CACHE / "log-data"




def validate_index(index_dir, source, verify_database=False):
    directory = Path(index_dir) / source
    meta = json.loads((directory / "index.json").read_text())
    if meta.get("status") != "complete" or meta.get("source") != source:
        raise ValueError("Incomplete or wrong-source log index")
    database = directory / "records.sqlite3"
    if not database.is_file():
        raise ValueError("Missing log database")
    if verify_database and file_sha256(database) != meta["sqlite_sha256"]:
        raise ValueError("Log database changed")
    return meta


def prepare_source(data_dir, source, index_dir=DEFAULT_INDEX):
    data_dir, directory = Path(data_dir), Path(index_dir) / source
    directory.mkdir(parents=True, exist_ok=True)
    source_file = data_dir / (source + "_labels.jsonl.gz")
    identity = {"source_file_sha256": file_sha256(source_file),
                "taxonomy_sha256": file_sha256(data_dir / "taxonomy.csv")}
    manifest = directory / "index.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if meta.get("status") == "complete":
            if any(meta.get(key) != value for key, value in identity.items()):
                raise ValueError("Existing index belongs to different data or taxonomy")
            return validate_index(index_dir, source)
    expected = int(source_inventory(data_dir)[source]["Logs"])
    categories = taxonomy(data_dir)
    meta = {"status": "preparing", "source": source, "partition": "all", "rows": expected,
            "source_file": str(source_file.resolve()), **identity,
            "data_adaptation": "Original Content, spans and LineId stored unchanged; no train/test split"}
    atomic_json(manifest, meta)
    partial = directory / "records.partial.sqlite3"
    connection = sqlite3.connect(partial)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS records (line_id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        committed = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        buffered, count = [], 0
        with gzip.open(source_file, "rt", encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                if count <= committed:
                    continue
                row = json.loads(line)
                if row["LineId"] != count:
                    raise ValueError("Nonconsecutive original LineId")
                span_set(row["Content"], row["Labels"], categories)
                buffered.append((count, line.rstrip("\n")))
                if len(buffered) == 10000:
                    connection.executemany("INSERT INTO records VALUES (?,?)", buffered)
                    connection.commit()
                    buffered.clear()
        if buffered:
            connection.executemany("INSERT INTO records VALUES (?,?)", buffered)
        connection.commit()
        if count != expected or connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] != expected:
            raise ValueError("Source was not indexed completely")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    database = directory / "records.sqlite3"
    os.replace(partial, database)
    meta.update(status="complete", sqlite_sha256=file_sha256(database))
    atomic_json(manifest, meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX))
    parser.add_argument("--sources", nargs="+")
    args = parser.parse_args()
    for source in args.sources or source_inventory(args.data_dir):
        result = prepare_source(args.data_dir, source, args.index_dir)
        print(json.dumps({"source": source, "status": result["status"], "rows": result["rows"]}), flush=True)


if __name__ == "__main__":
    main()
