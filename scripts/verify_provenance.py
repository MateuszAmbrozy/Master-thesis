"""Verify local third-party hashes without requiring network access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    pibt = json.loads((ROOT/'third_party/pibt_provenance.json').read_text())
    for item in pibt['files']:
        path = ROOT/'src/primal_mapf/solvers/pypibt'/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['local_sha256']:
            raise ValueError(f'PIBT source changed: {path}')
        if item['local_sha256'] != item['upstream_sha256']:
            raise ValueError(f'PIBT upstream identity is not established: {path}')
    if hashlib.sha256((ROOT/'third_party/licenses/PIBT.txt').read_bytes()).hexdigest() != pibt['license_sha256']:
        raise ValueError('PIBT license checksum mismatch')
    primal = json.loads((ROOT/'third_party/primal_provenance.json').read_text())
    license_path = ROOT/'third_party/licenses/PRIMAL-upstream.txt'
    if hashlib.sha256(license_path.read_bytes()).hexdigest() != primal['license_sha256']:
        raise ValueError('PRIMAL upstream license checksum mismatch')
    maps = json.loads((ROOT/'maps/provenance.json').read_text())
    known = {}
    for item in maps['files']:
        path = ROOT/'maps'/item['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['local_sha256'] or not item['normalized_lines_equal']:
            raise ValueError(f'Map provenance mismatch: {path}')
        known[path.name] = path.read_text().splitlines()
    archive_maps = list((ROOT/'research/historical').rglob('*.map'))
    for path in archive_maps:
        if path.name not in known or path.read_text().splitlines() != known[path.name]:
            raise ValueError(f'Unverified archived map: {path}')
    print(f'Verified PIBT sources, PRIMAL upstream license, {len(known)} maps and {len(archive_maps)} archived map copies.')


if __name__ == '__main__':
    main()
