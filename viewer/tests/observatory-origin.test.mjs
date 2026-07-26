import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_OBSERVATORY_ORIGIN,
  observatoryWebSocketUrl,
  resolveObservatoryOrigin,
} from "../app/observatory-origin.mjs";

test("uses the production loopback default and derives the WebSocket URL", () => {
  assert.equal(resolveObservatoryOrigin(undefined), DEFAULT_OBSERVATORY_ORIGIN);
  assert.equal(
    observatoryWebSocketUrl("http://127.0.0.1:8766"),
    "ws://127.0.0.1:8766/live",
  );
});

test("accepts only path-free HTTP loopback origins", () => {
  for (const value of [
    "https://127.0.0.1:8766",
    "http://192.168.1.3:8766",
    "http://example.test:8766",
    "http://127.0.0.1:8766/api",
    "http://user:password@127.0.0.1:8766",
  ]) {
    assert.throws(() => resolveObservatoryOrigin(value), /loopback/);
  }
  assert.equal(
    resolveObservatoryOrigin("http://localhost:8766"),
    "http://localhost:8766",
  );
  assert.equal(
    resolveObservatoryOrigin("http://[::1]:8766"),
    "http://[::1]:8766",
  );
});
