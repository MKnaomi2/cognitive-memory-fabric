const DEFAULT_OBSERVATORY_ORIGIN = "http://127.0.0.1:8765";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

/**
 * Resolve the loopback Observatory API origin. The environment value is used
 * by engineering validation; production retains the established default.
 *
 * @param {string | undefined} configured
 * @returns {string}
 */
export function resolveObservatoryOrigin(configured) {
  const candidate = configured?.trim() || DEFAULT_OBSERVATORY_ORIGIN;
  let parsed;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error("Observatory API origin must be an absolute loopback URL");
  }
  if (
    parsed.protocol !== "http:" ||
    !LOOPBACK_HOSTS.has(parsed.hostname) ||
    parsed.username ||
    parsed.password ||
    parsed.pathname !== "/" ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error(
      "Observatory API origin must be an HTTP loopback origin without credentials or a path",
    );
  }
  return parsed.origin;
}

/**
 * @param {string} apiOrigin
 * @returns {string}
 */
export function observatoryWebSocketUrl(apiOrigin) {
  const parsed = new URL(apiOrigin);
  parsed.protocol = "ws:";
  parsed.pathname = "/live";
  return parsed.toString();
}

export { DEFAULT_OBSERVATORY_ORIGIN };
