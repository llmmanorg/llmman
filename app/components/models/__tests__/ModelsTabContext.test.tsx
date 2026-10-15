// The Models tab's pull, against a fake daemon: success with notices, the
// daemon's error, Stop watching, and watching again after the OS dropped
// the connection in the background.

import { act, renderHook, waitFor } from "@testing-library/react-native";
import type { ReactNode } from "react";
import { AppState, type AppStateStatus } from "react-native";

import { ApiClient, type RequestInitLike, type ResponseLike } from "@/api/client";
import { fakeFetch, fixture, response } from "@/test/fakeFetch";
import { ModelsTabProvider, useModelsTab } from "../ModelsTabContext";

const URL = "http://10.0.0.5:17434";
const REF = "hf.co/QuantFactory/SmolLM2-135M-Instruct-GGUF:Q2_K";
const NOTICE =
  "warning: cannot verify hf.co/QuantFactory/SmolLM2-135M-Instruct-GGUF:Q2_K: no signature found";
const mockRefreshChat = jest.fn(async () => {});
let mockClient: ApiClient;

jest.mock("@/lib/ServersContext", () => ({ useServers: () => ({ client: mockClient }) }));
jest.mock("@/lib/ModelsContext", () => ({ useModels: () => ({ refresh: mockRefreshChat }) }));

// AppState, driven by the test.
let appState: AppStateStatus = "active";
const appListeners = new Set<(s: AppStateStatus) => void>();
function setAppState(s: AppStateStatus) {
  appState = s;
  for (const l of [...appListeners]) l(s);
}

beforeEach(() => {
  appState = "active";
  appListeners.clear();
  mockRefreshChat.mockClear();
  Object.defineProperty(AppState, "currentState", { configurable: true, get: () => appState });
  jest.spyOn(AppState, "addEventListener").mockImplementation((_type, l) => {
    appListeners.add(l as (s: AppStateStatus) => void);
    return { remove: () => appListeners.delete(l as (s: AppStateStatus) => void) } as never;
  });
});

afterEach(() => jest.restoreAllMocks());

/** A daemon answering the store routes from fixtures and `/api/pull` with `pull()`. */
function daemon(pull: (init: RequestInitLike) => ResponseLike) {
  const { fetch, calls } = fakeFetch((url, init) => {
    const path = url.slice(URL.length + 1);
    if (path === "v1/models") return response(fixture("v1-models.json"));
    if (path === "api/tags") return response(fixture("tags.json"));
    if (path === "llmman/node") return response(fixture("node.json"));
    if (path === "api/pull") return pull(init);
    return response('{"error":"not found"}', { status: 404 });
  });
  mockClient = new ApiClient({ url: URL }, fetch);
  return { pulls: () => calls.filter((c) => c.url.endsWith("/api/pull")) };
}

/** A body that sends `text` and then fails, as a connection the OS closed. */
function dropped(text: string): ResponseLike {
  const bytes = new TextEncoder().encode(text);
  let sent = false;
  return {
    ...response(""),
    body: new ReadableStream<Uint8Array>({
      pull(controller) {
        if (!sent) {
          sent = true;
          controller.enqueue(bytes);
        } else controller.error(new Error("Software caused connection abort"));
      },
    }),
  };
}

/** A body that never ends until the request is aborted. */
function forever(init: RequestInitLike): ResponseLike {
  return {
    ...response(""),
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"status":"pulling manifest"}\n'));
        init.signal?.addEventListener("abort", () =>
          controller.error(Object.assign(new Error("aborted"), { name: "AbortError" })),
        );
      },
    }),
  };
}

const wrapper = ({ children }: { children: ReactNode }) => <ModelsTabProvider>{children}</ModelsTabProvider>;

/** Lets the provider's follow-up work (the store re-read after a pull) land inside act. */
async function settle() {
  await act(() => new Promise((r) => setTimeout(r, 30)));
}

async function mounted() {
  const hook = renderHook(() => useModelsTab(), { wrapper });
  await waitFor(() => expect(hook.result.current.localReady).toBe(true));
  return hook;
}

it("reads the store, the details and the memory", async () => {
  daemon(() => response(""));
  const { result } = await mounted();
  expect(result.current.local.map((m) => m.id)).toContain("docker.io/ai/qwen3.5:0.8b");
  expect(result.current.details.get("docker.io/ai/qwen3.5:0.8b")?.size).toBe(737504352);
  expect(result.current.memory).toBe(68719476736);
});

it("follows a pull to success, keeps its notices, and refreshes the chat's list", async () => {
  // The recorded pull, with a verification notice before its last line as a daemon with a signature policy sends.
  const recorded = fixture("pull-hf-q2k.ndjson").replace(
    '{"status":"success"}',
    `{"notice":"${NOTICE}"}\n{"status":"success"}`,
  );
  const { pulls } = daemon(() => response(recorded, { chunk: 13 }));
  const { result } = await mounted();
  act(() => void result.current.startPull(REF));
  expect(result.current.startPull("another")).toBe(false); // one at a time

  await waitFor(() => expect(result.current.pull?.done).toBe(true));
  expect(result.current.pull).toMatchObject({ ref: REF, total: 88201792, completed: 88201792, error: null });
  expect(pulls()).toHaveLength(1);
  expect(JSON.parse(pulls()[0]!.init.body!)).toEqual({ model: REF, stream: true });
  await waitFor(() => expect(mockRefreshChat).toHaveBeenCalled());
  expect(result.current.noticesFor("hf.co/QuantFactory/SmolLM2-135M-Instruct-GGUF")).toEqual([
    {
      ref: REF,
      notices: [NOTICE],
    },
  ]);
  expect(result.current.noticesFor("hf.co/other/repo")).toEqual([]);
  await settle();
});

it("shows the daemon's error and frees the slot", async () => {
  daemon(() => response(fixture("pull-error.ndjson")));
  const { result } = await mounted();
  act(() => void result.current.startPull(REF));
  await waitFor(() =>
    expect(result.current.pull?.error?.message).toMatch(/^HF model info: GET https:\/\/huggingface\.co/),
  );
  expect(result.current.pull?.done).toBe(false);
  act(() => result.current.dismissPull());
  expect(result.current.pull).toBeNull();
  let again = false;
  act(() => void (again = result.current.startPull(REF)));
  expect(again).toBe(true);
  await settle();
});

it("stops watching without an error", async () => {
  daemon(forever);
  const { result } = await mounted();
  act(() => void result.current.startPull(REF));
  await waitFor(() => expect(result.current.pull?.status).toBe("pulling manifest"));
  act(() => result.current.stopWatching());
  await waitFor(() => expect(result.current.pull).toBeNull());
  let again = false;
  act(() => void (again = result.current.startPull(REF)));
  expect(again).toBe(true);
  await settle();
});

it("watches the same pull again when the app comes back after the OS dropped it", async () => {
  let attempt = 0;
  const { pulls } = daemon(() => {
    attempt += 1;
    return attempt === 1
      ? dropped('{"status":"pulling manifest"}\n{"status":"pulling","total":100,"completed":40}\n')
      : response('{"status":"pulling","total":100,"completed":100}\n{"status":"success"}\n');
  });
  const { result } = await mounted();
  act(() => void result.current.startPull(REF));
  act(() => setAppState("background"));

  await waitFor(() => expect(pulls()).toHaveLength(1));
  // Still in the background: no second request, and no error shown.
  await settle();
  expect(pulls()).toHaveLength(1);
  expect(result.current.pull?.error).toBeNull();
  expect(result.current.pull?.completed).toBe(40);

  act(() => setAppState("active"));
  await waitFor(() => expect(result.current.pull?.done).toBe(true));
  expect(pulls()).toHaveLength(2);
  expect(JSON.parse(pulls()[1]!.init.body!)).toEqual({ model: REF, stream: true });
  await settle();
});

it("reports a dropped connection in the foreground as an error", async () => {
  daemon(() => dropped('{"status":"pulling manifest"}\n'));
  const { result } = await mounted();
  act(() => void result.current.startPull(REF));
  await waitFor(() => expect(result.current.pull?.error?.message).toMatch(/^the connection dropped/));
  await settle();
});
