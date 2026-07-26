import { defineConfig } from "@playwright/test";
import path from "node:path";

const apiOrigin = "http://127.0.0.1:8766";
const viewerOrigin = "http://localhost:5173";
const fixturePython = process.env.CMF_E2E_PYTHON ?? "python";
const fixtureRoot =
  process.env.CMF_E2E_ROOT ??
  path.resolve("test-results", `real-backend-${process.pid}`);

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: viewerOrigin,
    browserName: "chromium",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `"${fixturePython}" ../tests/observatory_fixture.py`,
      url: `${apiOrigin}/health`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        ...process.env,
        CMF_E2E_ROOT: fixtureRoot,
        CMF_E2E_API_PORT: "8766",
      },
    },
    {
      command: "npm run dev -- --host localhost --port 5173",
      url: viewerOrigin,
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        ...process.env,
        CMF_VIEWER_PORT: "5173",
        NEXT_PUBLIC_OBSERVATORY_API_ORIGIN: apiOrigin,
      },
    },
  ],
});
