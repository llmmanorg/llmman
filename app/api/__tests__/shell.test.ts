import { ApiClient } from "@/api/client";
import { NO_SHELL_ENDPOINT, shellSocketUrl, shellStatus } from "@/api/shell";
import { keyProtocol } from "@/api/socket";
import { fakeFetch, llmman401, response } from "@/test/fakeFetch";

const URL = "http://127.0.0.1:17492";

test("reads the daemon's {enabled, reason}", async () => {
  const on = fakeFetch(() => response('{"enabled":true}'));
  expect(await shellStatus(new ApiClient({ url: URL }, on.fetch))).toEqual({ enabled: true, endpoint: true });
  expect(on.calls[0]!.url).toBe(`${URL}/llmman/shell`);

  const reason =
    "llmman serve is not bound to loopback (LLMMAN_HOST), so the shell is off: anyone who can reach the daemon could reach the shell";
  const off = fakeFetch(() => response(JSON.stringify({ enabled: false, reason })));
  expect(await shellStatus(new ApiClient({ url: URL, key: "k" }, off.fetch))).toEqual({
    enabled: false,
    endpoint: true,
    reason,
  });
});

test("a 404 is a daemon with no shell endpoint", async () => {
  const { fetch } = fakeFetch(() => response('{"error":"not found"}', { status: 404 }));
  expect(await shellStatus(new ApiClient({ url: URL }, fetch))).toEqual({
    enabled: false,
    endpoint: false,
    reason: NO_SHELL_ENDPOINT,
  });
});

test("other failures are errors", async () => {
  const { fetch } = fakeFetch(() => llmman401());
  await expect(shellStatus(new ApiClient({ url: URL }, fetch))).rejects.toMatchObject({ status: 401 });
});

test("the WebSocket address keeps a path prefix", () => {
  expect(shellSocketUrl(URL)).toBe("ws://127.0.0.1:17492/llmman/shell");
  expect(shellSocketUrl("https://gw.example/a")).toBe("wss://gw.example/a/llmman/shell");
  expect(keyProtocol("sk-test")).toMatch(/^llmman\.bearer\.[A-Za-z0-9_-]+$/);
});
