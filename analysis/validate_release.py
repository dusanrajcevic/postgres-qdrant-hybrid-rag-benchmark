from __future__ import annotations

import csv
import json
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = ROOT / "results" / "measurements"
SIZES = (100000, 250000)
ENGINES = ("pg-hnsw", "pg-diskann", "qdrant")
WORKLOADS = ("w1", "w2_50", "w2_10", "w2_1", "w2_0_1", "w3", "w4_acl")
EXPECTED_RANKS = set(range(100, 1000))
EXPECTED_ROWS_PER_REP = len(WORKLOADS) * len(EXPECTED_RANKS)
CHECKSUM_FILE = ROOT / "checksums" / "ARTIFACTS.sha256"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_rep(path: Path, engine: str, rep: int, size: int) -> None:
    rows = read_rows(path)
    assert len(rows) == EXPECTED_ROWS_PER_REP, (path, len(rows), EXPECTED_ROWS_PER_REP)

    by_workload: dict[str, list[dict[str, str]]] = {w: [] for w in WORKLOADS}
    for row in rows:
        assert row["engine"] == engine, (path, row["engine"], engine)
        assert int(row["dataset_size"]) == size, (path, row["dataset_size"], size)
        assert int(row["repetition"]) == rep, (path, row["repetition"], rep)
        workload = row["workload"]
        assert workload in by_workload, (path, workload)
        by_workload[workload].append(row)
        assert int(row["returned_count"]) == 10, (path, workload, row["query_rank"], row["returned_count"])

    for workload, wr in by_workload.items():
        assert len(wr) == 900, (path, workload, len(wr))
        ranks = [int(r["query_rank"]) for r in wr]
        assert set(ranks) == EXPECTED_RANKS, (path, workload, min(ranks), max(ranks))
        assert len(ranks) == len(set(ranks)), (path, workload, "duplicate query ranks")


def validate_summary(path: Path, size: int) -> None:
    rows = read_rows(path)
    assert len(rows) == len(ENGINES) * len(WORKLOADS), (path, len(rows))
    keys = Counter((r["engine"], r["workload"]) for r in rows)
    assert all(v == 1 for v in keys.values()), (path, "duplicate summary rows")
    assert set(keys) == {(e, w) for e in ENGINES for w in WORKLOADS}, (path, "missing summary rows")
    for row in rows:
        assert int(row["dataset_size"]) == size
        assert int(row["observations"]) == 2700
        assert int(row["unique_queries_per_repetition"]) == 900
        assert int(row["repetitions"]) == 3
        assert int(row["min_returned"]) == 10
        if "queries_with_fewer_than_10_pooled" in row and row["queries_with_fewer_than_10_pooled"]:
            assert int(row["queries_with_fewer_than_10_pooled"]) == 0



TEXT_CHECKSUM_SUFFIXES = {".csv", ".json"}


def checksum_bytes(path: Path) -> bytes:
    """Return stable bytes for release checksums across Git checkouts.

    Git may normalize line endings for text files according to .gitattributes.
    The scientific content of CSV and JSON artifacts is unchanged by CRLF/LF
    conversion, so checksums use LF as the canonical line ending for those
    text formats.
    """
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_CHECKSUM_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def validate_checksums() -> int:
    assert CHECKSUM_FILE.exists(), CHECKSUM_FILE
    checked = 0
    for line_number, line in enumerate(
        CHECKSUM_FILE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, rel = line.split("  ", 1)
        except ValueError as exc:
            raise AssertionError(
                f"Malformed checksum entry on line {line_number}: {line!r}"
            ) from exc

        path = ROOT / rel
        assert path.exists(), path
        digest = hashlib.sha256(checksum_bytes(path)).hexdigest()
        assert digest == expected, (path, digest, expected)
        checked += 1
    return checked


def main() -> None:
    checksum_count = validate_checksums()
    checked = 0
    for size in SIZES:
        directory = MEASUREMENTS / str(size)
        assert directory.is_dir(), directory
        manifest = directory / "measurement_manifest.json"
        assert manifest.exists(), manifest
        json.loads(manifest.read_text(encoding="utf-8"))

        for engine in ENGINES:
            for rep in (1, 2, 3):
                path = directory / f"{engine}_rep{rep}.csv"
                assert path.exists(), path
                validate_rep(path, engine, rep, size)
                checked += 1

        summary = directory / f"final_summary_{size}.csv"
        assert summary.exists(), summary
        validate_summary(summary, size)

    print(f"Release validation passed: {checked} repetition CSVs, 113,400 held-out observations total.")
    print(f"Verified SHA-256 checksums for {checksum_count} frozen configuration and measurement artifacts.")
    print("All workloads contain ranks 100 to 999 exactly once per repetition and all requests returned 10 results.")


if __name__ == "__main__":
    main()
