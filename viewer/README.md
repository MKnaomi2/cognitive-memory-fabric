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
port 8765. Run `npm test`, `npm run lint`, and `npm audit` before publishing.
