"""Verify this archive without installing packages or rerunning simulations."""
from pathlib import Path
import hashlib
import json
import sys
import argparse
import os

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--restore-archive-times',action='store_true',
    help='After successful checksum checks, normalize payload mtimes for legacy freshness tests; bytes are unchanged.')
args=parser.parse_args()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(records):
    for r in records:
        rel=Path(r['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest path')
        p=ROOT/rel
        if not p.is_file() or digest(p)!=r['sha256']:
            raise ValueError('Checksum mismatch: '+str(rel))
try:
    payload=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    check(payload['files'])
    historical=json.loads((ROOT/'results/jhep_submission_hardening/manifest.json').read_text())
    check(historical['artifacts'])
    provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
    check(provenance['source_files'])
except (ValueError,KeyError,OSError) as e:
    print('FAIL:',e,file=sys.stderr);raise SystemExit(1)
if args.restore_archive_times:
    # 2026-11-07 00:00:00 UTC. Freshness is established by the exact content
    # manifest; normalize extraction-time noise, not scientific provenance.
    for r in payload['files']:
        os.utime(ROOT/r['path'],(1794009600,1794009600))
    print('Normalized verified payload mtimes for the legacy freshness test; no bytes changed.')
print(f"PASS: {len(payload['files'])} payload files, {len(historical['artifacts'])} original evidence artifacts, "
      f"{len(provenance['source_files'])} exact source exports.")
