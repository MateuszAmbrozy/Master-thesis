# Sources, attribution, and distribution records

The root [MIT license](LICENSE) applies to this project's original contributions.
It does not replace the licenses of incorporated third-party software or data.
The records below distinguish verified upstream sources from unresolved history.

## PRIMAL

The classic network and grid-environment design are local adaptations of
[PRIMAL](https://github.com/gsartoretti/PRIMAL), whose README credits
Guillaume Sartoretti and Justin Kerr. The repository describes a PyTorch
adaptation; it does not claim that PRIMAL or its original architecture was
invented here.

The original local license copy is preserved in
[PRIMAL.txt](third_party/licenses/PRIMAL.txt). An independently downloaded
upstream license is preserved in
[PRIMAL-upstream.txt](third_party/licenses/PRIMAL-upstream.txt), with its source
revision and checksum in [the provenance record](third_party/primal_provenance.json).
The license's existing copyright text is retained verbatim. The precise upstream
revision from which the historical local adaptation was made is unknown.

## PIBT

The vendored Python implementation originates from
[Keisuke Okumura's pypibt](https://github.com/Kei18/pypibt).
The three implementation files were verified byte-for-byte against revision
`a3c97f60413c6619a29a5022969896bc54877edc`.

The original MIT license, copyright 2023 Keisuke Okumura, is included in
[PIBT.txt](third_party/licenses/PIBT.txt). Source URLs and SHA-256 hashes are
recorded in [pibt_provenance.json](third_party/pibt_provenance.json).
The algorithm and implementation are third-party contributions.

## Cooperative A* and CBS

The C++ implementations in `native/ca` and `native/cbs` were written by
**Mateusz Ambroży** for this project and are covered by the root
[MIT license](LICENSE). Their authorship was confirmed by the project author.
Other MAPF repositories were consulted as implementation references; the
specific repository URLs were not recorded.

The original implementation notices are retained. PRIMAL, PIBT, and benchmark
data have separate attribution and license records below and above.

## Maps

The three maps in `maps/` were compared with the official Moving AI downloads;
their text lines match. Provider URLs, hashes, and comparison results are in
[maps/provenance.json](maps/provenance.json). Nine map copies inside the historical
archive use the same filenames and are checked against those verified map texts
by the local provenance checker.

Moving AI identifies its benchmark data as available under the
[Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).
Attribution: **Moving AI Lab, Nathan Sturtevant**. See the
[benchmark collection](https://movingai.com/benchmarks/) and
[grid benchmark documentation](https://movingai.com/benchmarks/grids.html).
When using these benchmarks, cite:

> Sturtevant, N. (2012). Benchmarks for Grid-Based Pathfinding.
> IEEE Transactions on Computational Intelligence and AI in Games, 4(2), 144–148.

Map contents are retained; generated starts and goals are new instances and do
not reproduce official scenario files. The maps are not relicensed under this
project's MIT license.

## Historical research materials

Historical scripts, measurements, and figures were imported from the existing
local research folders. Source paths, transformations, and checksums are retained
in [the research manifest](research/historical/file_manifest.json).
Historical measurements are not results of the current trainer.

Run `python scripts/verify_provenance.py` to check recorded local hashes and
archived map identity. This checks file identity against the recorded source hashes.
