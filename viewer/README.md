# Hermes Neural Observatory

Local, read-only WebGPU/WebGL visualization for Cognitive Memory Fabric's
Hippocampal Replay Engine.
It renders the EC→DG→CA3→CA1 topology, live MessagePack telemetry, provenance
inspection, and recorded `.hmrec` sleep replay.

```powershell
npm ci
npm run dev
```

The UI listens on loopback port 3000 and reads the loopback telemetry API on
port 8765. Engineering validation may set
`NEXT_PUBLIC_OBSERVATORY_API_ORIGIN=http://127.0.0.1:8766`; non-loopback,
credential-bearing, HTTPS, and path-bearing origins are rejected. Run
`npm test`, `npm run e2e`, `npm run lint`, and `npm audit` before publishing.
The E2E command starts the real Python API on port 8766 and the viewer on port
5173 with SQLite, MessagePack WebSocket frames, and an `.hmrec` recording.
