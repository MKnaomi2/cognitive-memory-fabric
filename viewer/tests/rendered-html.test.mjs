import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the neural observatory shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Hermes Neural Observatory<\/title>/i);
  assert.match(html, /Neural Observatory/);
  assert.match(html, /Entorhinal cortex/);
  assert.match(html, /Dentate gyrus/);
  assert.match(html, /CA3 recurrent field/);
  assert.match(html, /CA1 output/);
  assert.match(html, /Replay timeline/);
  assert.match(html, /read-only/);
  assert.match(html, /No raw transcripts are sent to this viewer/);
});

test("keeps rendering, telemetry, and replay controls local", async () => {
  const [page, observatory, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/NeuralObservatory.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /<NeuralObservatory \/>/);
  assert.match(page, /Hermes Neural Observatory/);
  assert.match(observatory, /http:\/\/127\.0\.0\.1:8765/);
  assert.match(observatory, /ws:\/\/127\.0\.0\.1:8765\/live/);
  assert.match(observatory, /WebGPURenderer/);
  assert.match(observatory, /WebGLRenderer/);
  assert.match(observatory, /HMREC1\\n/);
  assert.match(observatory, /EC:\s*8192/);
  assert.match(observatory, /DG:\s*16384/);
  assert.match(observatory, /CA3:\s*8192/);
  assert.match(observatory, /CA1:\s*4096/);
  assert.match(packageJson, /"three":/);
  assert.doesNotMatch(observatory, /https?:\/\/(?!127\.0\.0\.1)/);
});
