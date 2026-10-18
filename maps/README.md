# Maps

Moving AI benchmark maps used for fixed-map training and evaluation:

| File | Map family |
| --- | --- |
| `maze-32-32-4.map` | Maze |
| `room-64-64-16.map` | Rooms |
| `warehouse-10-20-10-2-1.map` | Warehouse |

Select a map in a training or evaluation configuration:

```json
{
  "map": "maps/maze-32-32-4.map"
}
```

The loader reads MovingAI `.map` files. `.` and `G` are traversable; other
symbols are obstacles. Starts and goals are sampled as distinct cells in the
largest connected free component, which must contain at least twice as many
cells as agents. The current sampler does not read `.scen` files.

Source URLs and file hashes are recorded in [provenance.json](provenance.json).
The maps match the official downloads, including the copies in the research
archive.

## Attribution

Maps: **Moving AI Lab, Nathan Sturtevant**, under the
[Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).
Please cite:

Sturtevant, N. (2012). *Benchmarks for Grid-Based Pathfinding*.
IEEE Transactions on Computational Intelligence and AI in Games, 4(2), 144–148.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for the benchmark source links.
