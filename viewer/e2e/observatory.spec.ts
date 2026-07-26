import { encode } from "@msgpack/msgpack";
import { expect, test } from "@playwright/test";
import type { APIRequestContext } from "@playwright/test";

const API = "http://127.0.0.1:8766";
const TOKEN = "cmf-e2e-test-token-not-a-secret-" + "0".repeat(40);

async function ingest(
  request: APIRequestContext,
  step: number,
) {
  const payload = encode({
    step,
    phase: "rem",
    active_neurons: [0, 8192, 24576],
    active_edges: [[0, 8192, 0.75, "EC_DG"]],
    region_spikes: { EC: 1, DG: 1, CA3: 1, CA1: 0 },
  });
  const response = await request.post(`${API}/ingest`, {
    data: Buffer.from(payload),
    headers: {
      authorization: `Bearer ${TOKEN}`,
      "content-type": "application/msgpack",
    },
  });
  expect(response.ok()).toBeTruthy();
}

test("drives the real loopback Observatory and recording", async ({
  page,
  request,
}) => {
  await page.goto("/");

  const guide = page.getByRole("dialog", {
    name: "How to read the neural observatory",
  });
  await expect(guide).toBeVisible();
  await guide.getByRole("button", { name: "Understood" }).click();
  await page.reload();
  await expect(guide).toBeHidden();

  await expect(page.getByText("36,864", { exact: true })).toBeVisible();
  await expect(page.getByText("770,048", { exact: true })).toBeVisible();
  await expect(page.getByText("Live circuit", { exact: true })).toBeVisible();
  await expect(page.getByText(/trisynaptic-v3-content-readout/)).toBeVisible();

  await ingest(request, 4242);
  await expect(page.getByText("Step 4,242", { exact: true })).toBeVisible();
  await expect(page.getByText("3", { exact: true }).last()).toBeVisible();

  await page.getByRole("button", { name: "Pause" }).click();
  await ingest(request, 4243);
  await expect(page.getByText("Step 4,242", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Play" }).click();
  await ingest(request, 4244);
  await expect(page.getByText("Step 4,244", { exact: true })).toBeVisible();

  const recordings = await (await request.get(`${API}/recordings`)).json();
  const recording = recordings.recordings[0].name as string;
  await page.locator("footer select").selectOption(recording);
  await expect(page.getByText("Step 9,002", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Replay timeline")).toHaveValue("100");
  await expect(page.getByText("NREM", { exact: true })).toBeVisible();

  const neuron = await (await request.get(`${API}/neuron/0`)).json();
  expect(neuron.neuron_id).toBe(0);
  expect(neuron.incoming).toBeInstanceOf(Array);
  expect(neuron.outgoing).toBeInstanceOf(Array);
});
