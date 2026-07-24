# Observability

## Purpose

The Neural Observatory makes memory replay inspectable without giving the
visualization layer lifecycle authority. It combines static circuit geometry,
binary live telemetry, recorded sleep sessions, and bounded provenance views.

## Network boundary

The FastAPI service accepts only `127.0.0.1` or `::1` as its bind address.
Allowed browser origins are loopback ports 3000 and 5173. API documentation
routes are disabled.

| Route | Method | Authority |
|---|---|---|
| `/health` | GET | service/schema status |
| `/geometry` | GET | visual coordinates, region ranges, and authoritative pathway metadata |
| `/neuron/{id}` | GET | bounded exact incoming/outgoing adjacency from the live circuit |
| `/snapshot` | GET | aggregate memory/conflict/event counts |
| `/memory/{id}` | GET | one memory's bounded metadata, evidence, events, and engram IDs |
| `/recordings` | GET | safe local recording names and byte sizes |
| `/recordings/{name}` | GET | one containment-checked `.hmrec` file |
| `/live` | WebSocket | binary live/backlog frames |
| `/ingest` | POST | authenticated telemetry publication only |

There is no lifecycle mutation route. `/ingest` cannot create, update, archive,
or consolidate a memory.

The publisher token is a 32-byte random value represented by 64 hex characters.
It is stored below the runtime directory and compared with constant-time HMAC
comparison. Ingest additionally requires a loopback client, a positive content
length, and a payload no larger than 16 MiB.

## Geometry schema

```json
{
  "schema": 2,
  "circuit_version": "trisynaptic-v3-content-readout",
  "neuron_count": 36864,
  "positions": [[-3.2, 0.4, 0.1]],
  "layout": {
    "kind": "illustrative-annular",
    "authority": "visual-only",
    "distance_semantics": false,
    "default_view": "functional-topology"
  },
  "regions": {
    "EC": {
      "start": 0,
      "end": 8192,
      "count": 8192,
      "role": "context and cortical input"
    }
  },
  "pathways": [{
    "name": "EC_DG",
    "source": "EC",
    "target": "DG",
    "fanout": 8,
    "synapse_count": 65536,
    "inhibitory": false,
    "plastic": true,
    "recurrent": false
  }]
}
```

Coordinates are deterministic visual scaffolding. Their ring shape, distance,
and overlap have no anatomical or learned meaning. The source/target pathway
graph and its synapse counts describe the implemented topology. Static arrows
are aggregate summaries; only `active_edges` and `/neuron/{id}` expose exact
edges. Geometry is separated from live frames so coordinates transfer once.

## Telemetry frame

Frames are MessagePack maps:

```json
{
  "schema": 1,
  "step": 150,
  "phase": "rem",
  "memory_id": 42,
  "active_neurons": [24580, 33104],
  "active_edges": [[24580, 33104, 0.5921, "CA3_CA1"]],
  "region_spikes": {"EC": 0, "DG": 0, "CA3": 1, "CA1": 1}
}
```

The hub retains the most recent 600 encoded frames. New WebSocket clients
receive this backlog before subsequent live frames. Failed clients are removed
without interrupting the simulation.

## Recording format

`.hmrec` is intentionally simple and streamable:

```text
7 bytes     ASCII "HMREC1\n"
4 bytes     little-endian unsigned frame length
N bytes     MessagePack frame
...         repeated
```

The reader rejects a wrong magic value, truncated header, frame above 16 MiB,
or truncated payload. The writer defaults to a 512 MiB total bound.

## 3-D viewer

The React/Three.js viewer:

- prefers `WebGPURenderer` and falls back to `WebGLRenderer`;
- defaults to a functional layout with separated region volumes;
- labels the coordinates as illustrative rather than anatomical;
- offers the original annular coordinates only as an explicitly illustrative mode;
- renders configured source/target pathways as aggregate arrows and recurrent loops;
- distinguishes those summaries from exact measured active edges;
- uses region clusters as distance-based level of detail;
- colors EC, DG, CA3, and CA1 independently;
- overlays bounded active pathways;
- can include/exclude inhibitory activity;
- supports orbit, pan, zoom, and WASD/QE free flight;
- ray-picks neurons and requests bounded exact adjacency for inspection;
- shows a pathway connection matrix with fanout and synapse totals;
- provides a first-use guide explaining what is measured versus illustrative;
- displays current phase, circuit step, per-region spikes, and active count; and
- reads `.hmrec` recordings into a scrub/play timeline.

The viewer never sends raw transcripts. The per-memory endpoint removes the
structured provenance JSON and holographic vector, returning source/evidence
metadata and event hashes instead.

## Local persistence

By default:

```text
D:\HermesMemory\neural\recordings\sleep-<session>.hmrec
D:\HermesMemory\neural\checkpoints\checkpoint-<session>.pt
%LOCALAPPDATA%\hermes\runtime\observatory.token
%LOCALAPPDATA%\hermes\logs\neural-observatory\
```

These paths are runtime artifacts and are excluded from source control.
