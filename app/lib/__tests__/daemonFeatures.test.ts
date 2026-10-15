import { ApiClient } from "@/api/client";
import { detectFeatures } from "@/lib/daemonFeatures";
import { fakeFetch, response } from "@/test/fakeFetch";

const URL = "http://127.0.0.1:17492";
const notFound = () => response('{"error":"not found"}', { status: 404 });

function daemon(routes: Record<string, () => ReturnType<typeof response>>) {
  return fakeFetch((url) => {
    const path = url.slice(URL.length + 1);
    return (routes[path] ?? notFound)();
  });
}

test("on loopback: a shell", async () => {
  const { fetch } = daemon({ "llmman/shell": () => response('{"enabled":true}') });
  expect(await detectFeatures(new ApiClient({ url: URL }, fetch))).toEqual({
    shell: { enabled: true, endpoint: true },
  });
});

test("beyond loopback: the shell off, with the daemon's reason", async () => {
  const { fetch } = daemon({
    "llmman/shell": () => response('{"enabled":false,"reason":"llmman serve is not bound to loopback"}'),
  });
  const f = await detectFeatures(new ApiClient({ url: URL, key: "k" }, fetch));
  expect(f.shell).toEqual({
    enabled: false,
    endpoint: true,
    reason: "llmman serve is not bound to loopback",
  });
});

test("an older daemon, and failures, mean no shell", async () => {
  const old = daemon({});
  expect(await detectFeatures(new ApiClient({ url: URL }, old.fetch))).toMatchObject({
    shell: { enabled: false, endpoint: false },
  });

  const down = fakeFetch(() => {
    throw new Error("connection refused");
  });
  const f = await detectFeatures(new ApiClient({ url: URL }, down.fetch));
  expect(f.shell.enabled).toBe(false);
  expect(f.shell.reason).toMatch(/could not reach/);
});
