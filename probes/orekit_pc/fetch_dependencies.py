"""Fetch only the pinned public Maven artifacts and verify their exact bytes."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--deps", type=Path, required=True)
args = parser.parse_args()
args.deps.mkdir(parents=True, exist_ok=True)
for item in json.loads((Path(__file__).parent / "dependencies.json").read_text()):
    target = args.deps / item["filename"]
    if not target.exists():
        urllib.request.urlretrieve(item["url"], target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Dependency fingerprint mismatch: " + item["filename"])
print("Pinned dependencies verified.")
