# v0.5.1 Engineering Validation Status

Status: Windows engineering validation passed, pull request #6 merged, and all
post-merge `main` checks passed. This evidence supports a v0.5.1 engineering
prerelease; it does not establish neural superiority. The preregistered
1,000-trial publication study remains out of scope.

- Validation pull request: https://github.com/MKnaomi2/cognitive-memory-fabric/pull/6
- Post-merge CI: https://github.com/MKnaomi2/cognitive-memory-fabric/actions/runs/30189512704

## Tested revision and environment

- Tested commit: `af0ff3ba7a01e04d7821fee83fa59d89de7c6201`
- Repository state during evaluation reproduction: `dirty: false`
- OS: Windows 11 Pro `10.0.26200`
- Python: `3.11.9`
- Torch: `2.12.1+cu130`
- CUDA runtime: `13.0`
- GPU: NVIDIA GeForce RTX 5060 Ti
- NVIDIA driver: `596.49`
- Node: `24.15.0`
- npm: `11.12.1`
- Raw evidence: isolated `cmf-validation/20260726-011624` validation root

The validation worktree and every test data path passed the live-path overlap
check. The production checkout, Observatory, and Hermes scheduled-task
definitions were unchanged.

## Validation results

- Python: `48 passed, 0 skipped`
- Ruff: passed
- Python compilation: passed
- Viewer npm audit: zero vulnerabilities
- Viewer lint, build, and four Node tests: passed
- Playwright: one real-backend scenario passed on API port 8766 and viewer port
  5173
- Existing development and holdout artifacts: verified, including protocol and
  artifact hashes

The Playwright scenario used the real Python API, SQLite store, WebSocket,
MessagePack frames, and `.hmrec` recording. It covered first-use guide
persistence, real geometry, live connection, authenticated UI-changing
telemetry, pause/resume, recording selection, and final-frame scrubbing. Exact
neuron connectivity was verified through the real endpoint; canvas point
selection is not deterministic in the headless renderer.

## Circuit, migration, vault, and sleep

`circuit-check --device cuda` completed on CUDA with 36,864 neurons, 770,048
synapses, 4,733 engram neurons, and 25 time cells. Final regional spike counts
were EC 27, DG 14, CA3 13, and CA1 24.

Three synthetic memories were seeded with legacy engrams. The dry run identified
all three, and the isolated apply migrated all three to content-v3. Content
hashes, nonempty CA1 signatures, context/event/order bindings, and source
assessments were verified.

- Memory 1 SHA-256:
  `a8478440a9bdfe6dc1131ec1aa052a67e960ef02b7f940b60359849c3d7f251b`
- Memory 2 SHA-256:
  `27fb35cf69898fd1db97b0a329cc79439a4bc2d7552d7919022e2d6c20e01135`
- Memory 3 SHA-256:
  `09a762069693297b0b90b73dcd5eeb83dab830d5dc633341b1dfd89ab79b0a7b`

The bounded vault plan applied three mutations. The resulting staged vault had
10 notes and `VaultMigrator.audit.valid == true`, with zero invalid
frontmatter, duplicate IDs, duplicate groups, unresolved links, missing primary
maps, or missing relationships.

One lease-respecting bounded sleep pass completed, replayed three memories,
wrote 48 readable NREM/REM frames, and created three active time-cell bindings.
The checkpoint file and database registry both recorded SHA-256
`88049de4b8775889363d9553ff8cb72fce7bdf470a10a8d6833a7c1a30673b8e`.

## Frozen evaluation reproduction

Both splits used lexical cues, neural weight `0.05`, margin `0.0`, activation
`0.70`, and only `fabric-symbolic` and `fabric-neural`. Each run contained 56
worlds and 112 trials. The holdout rerun is reproducibility evidence only and
was not used for tuning or described as a new holdout.

| Split | Committed symbolic | Reproduced symbolic | Committed neural | Reproduced neural |
|---|---:|---:|---:|---:|
| Development | 0.982143 | 0.982143 | 1.000000 | 1.000000 |
| Holdout | 0.964286 | 0.964286 | 0.982143 | 0.982143 |

Development reproduction hashes:

- `dataset.json`: `2e0c67f9c245a7ecc596cb4ca1cc92d7bfc83e06ede05e1a1bd484f9a6b4e1da`
- `results.svg`: `db30a04e1ef826d0ebe46deb19000ea3b46edbed838e0b9b78fbd52a32eaae81`
- `summary.json`: committed
  `478f9f62d45a91355685389cac960f50a00609bc56fb17cd9169b20bc6568277`,
  reproduced
  `2bc1cb0090ceecbf457c482b66524325b1ed900d9137e41ab3fb028fdb6406a0`
- `trials.jsonl`: committed
  `ae35d7dd89c4d19da400bb35beb4363601740b5a8cc1e6e7391e3e220b8f8a27`,
  reproduced
  `ede73373cb08ea1ef683ae82b689a010d9c0097d8d17c7e530fccd09b6d83652`

Holdout reproduction hashes:

- `dataset.json`: `263d953ffed389a73afad66b149e0f9e8fc9603f0be4695dc737a9dcd503a12d`
- `results.svg`: `4f067cdc8a5782c04712f40aea80c5342a9486773e3b2456e1146dd4973921d2`
- `summary.json`: committed
  `e431fb0511293922a8deb3547654abbd15079a3450f6fb8da116c66f5f458be5`,
  reproduced
  `71e54c7fa3fdaf3ef561d8d1d74d1efce86b7e7e7ecb522444d82bc901c56180`
- `trials.jsonl`: committed
  `684731ad97975f2eef4e835f53cfb0ce6276c75ab18a0d153d45e99fdc43a344`,
  reproduced
  `3e2d97ee9f135da973281fd2efe8af6093e1552265cf4ffd55f1853f885f9c64`

Dataset and SVG hashes reproduced exactly. Outcome metrics reproduced exactly;
summary and trial byte hashes differ only in measured latency, CPU time, and
development storage-size fields.

## Confirmed fixes

- Enforced actual telemetry body size and valid content length, and converted
  malformed MessagePack into a bounded client error.
- Made the viewer API origin configurable while rejecting non-loopback,
  credential-bearing, HTTPS, and path-bearing origins.
- Added explicit Playwright dependency and real-backend browser/integration
  coverage.
- Updated vulnerable viewer dependencies; the high-severity audit gate now
  reports zero vulnerabilities.
- Normalized tracked protocol text hashes across LF and CRLF Git checkouts,
  with a regression assertion inside the unchanged 48-test count.

## Limitations

- Headless Chromium lacked WebGPU and used the viewer's WebGL2 fallback.
- The development and holdout reproductions are engineering reproducibility
  evidence, not publication evidence.
- No merge, tag, release, or publication-scale study was performed.
