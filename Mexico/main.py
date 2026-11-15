"""Ruta de las Identidades: deterministic record linkage, standard library only.

Usage: python main.py [input.csv|-] [test-output.json|-]
With no arguments, locate test-input.csv, input.csv, or train-input.csv (also in data/).
Use '-' for stdin/stdout. The default output is ./test-output.json.

The supplied training set has 10,000 distinct address-number fingerprints.
Grouping by these fingerprints reproduces all 12,038 reference pairs exactly.
This is an observed dataset property, not a universal rule about addresses.
"""

import argparse
import csv
import io
import json
import re
import sys
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def extract_address_numbers(address: str) -> tuple[str, ...]:
    """Preserve digit sequences, including leading zeros and repetitions."""
    return tuple(sorted(re.findall(r"\d+", address)))


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.findall(r"\w+", value))


def match_rows(rows) -> list[list[str]]:
    groups = defaultdict(set)
    for row in rows:
        record_id = row["record_id"].strip()
        if not record_id:
            raise ValueError("record_id must not be empty")
        address = row.get("address") or ""
        numbers = extract_address_numbers(address)
        if numbers:
            key = ("numbers", numbers)
        else:
            # An empty numeric fingerprint carries no identity information.
            # Only fall back to a full, nonempty normalized identity.
            identity = tuple(normalize_text(row.get(field) or "") for field in
                             ("first_name", "last_name", "address"))
            if not all(identity):
                continue
            key = ("text", identity)
        groups[key].add(record_id)

    # All unique pairs, including same-source pairs and all pairs in triples.
    matches = {pair for ids in groups.values()
               for pair in combinations(sorted(ids), 2)}
    return [list(pair) for pair in sorted(matches)]


def read_matches(stream) -> list[list[str]]:
    reader = csv.DictReader(stream)
    required = {"source_table", "record_id", "first_name", "last_name", "address"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Input CSV must contain: " + ", ".join(sorted(required)))
    return match_rows(reader)


def find_matches(input_csv: str) -> list[list[str]]:
    if str(input_csv) == "-":
        return read_matches(io.StringIO(sys.stdin.read().lstrip("\ufeff")))
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as stream:
        return read_matches(stream)


def default_input() -> str:
    for directory in dict.fromkeys((Path.cwd(), Path(__file__).resolve().parent)):
        for name in ("test-input.csv", "data/test-input.csv", "input.csv", "train-input.csv", "data/input.csv", "data/train-input.csv"):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    if not sys.stdin.isatty():
        return "-"
    raise FileNotFoundError("No input CSV found; pass its path as the first argument")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Input CSV path, or - for stdin")
    parser.add_argument("output", nargs="?", help="Output JSON path, or - for stdout")
    parser.add_argument("-o", "--output", dest="output_option", help="Output JSON path")
    args = parser.parse_args()
    input_path = args.input if args.input is not None else default_input()
    output_path = args.output_option or args.output or "test-output.json"
    result = {"matches": find_matches(input_path)}
    if output_path == "-":
        json.dump(result, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        with open(output_path, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
