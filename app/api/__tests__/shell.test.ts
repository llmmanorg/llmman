import { Platform } from "react-native";

import { ApiClient } from "@/api/client";
import {
  NO_SHELL_ENDPOINT,
  openShellSocket,
  shellSocketUrl,
  shellStatus,
  shellTokenProtocol,
} from "@/api/shell";
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

test("a daemon that needs the shell token says so; an older one does not", async () => {
  const current = fakeFetch(() => response('{"enabled":true,"token_required":true}'));
  expect(await shellStatus(new ApiClient({ url: URL }, current.fetch))).toEqual({
    enabled: true,
    endpoint: true,
    tokenRequired: true,
  });
  const older = fakeFetch(() => response('{"enabled":true}'));
  expect((await shellStatus(new ApiClient({ url: URL }, older.fetch))).tokenRequired).toBeUndefined();
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

function socketsOpened(os: typeof Platform.OS, open: () => void): unknown[][] {
  const opened: unknown[][] = [];
  const real = { WebSocket: globalThis.WebSocket, os: Platform.OS };
  globalThis.WebSocket = class {
    constructor(...args: unknown[]) {
      opened.push(args);
    }
  } as unknown as typeof WebSocket;
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
  try {
    open();
  } finally {
    globalThis.WebSocket = real.WebSocket;
    Object.defineProperty(Platform, "OS", { value: real.os, configurable: true });
  }
  return opened;
}

test("the shell token is offered as its own subprotocol, with the key as a header off the web", () => {
  expect(shellTokenProtocol("tok")).toBe("llmman.shell.tok");
  const opened = socketsOpened("ios", () => {
    openShellSocket(URL, "k", "tok");
    openShellSocket(URL, undefined, undefined);
  });
  expect(opened[0]).toEqual([
    "ws://127.0.0.1:17492/llmman/shell",
    ["llmman.shell.tok"],
    { headers: { Authorization: "Bearer k" } },
  ]);
  expect(opened[1]).toEqual(["ws://127.0.0.1:17492/llmman/shell", undefined, {}]);
});

test("in a browser, the key and the shell token are both offered as subprotocols", () => {
  const opened = socketsOpened("web", () => {
    openShellSocket(URL, "k", "tok");
    openShellSocket(URL, undefined, "tok");
  });
  expect(opened[0]).toEqual(["ws://127.0.0.1:17492/llmman/shell", [keyProtocol("k"), "llmman.shell.tok"]]);
  expect(opened[1]).toEqual(["ws://127.0.0.1:17492/llmman/shell", ["llmman.shell.tok"]]);
});
