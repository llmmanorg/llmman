/** @jest-environment node */

// Media generation turns in the chat store: the prompt and its result stay
// out of a chat's history, the result is stored with the chat, Stop and
// Retry behave as the web UI's do.

import AsyncStorage from "@react-native-async-storage/async-storage";

import { ApiClient, type ResponseLike } from "@/api/client";
import { fakeFetch, fixture, response } from "@/test/fakeFetch";
import {
  __resetChatStore,
  getConversationState as state,
  historyFor,
  openConversation,
  retry,
  send,
  setMediaOptions,
  stop,
  subscribe,
} from "../chatStore";
import { exportJson, loadConversation, newConversation, type Message } from "../conversations";
import { removeMedia, storeMedia } from "../mediaFiles";

jest.mock("expo-crypto", () => {
  let n = 0;
  return { randomUUID: () => `g-${++n}` };
});

// The files themselves are a device's business; here, where they would go.
jest.mock("../mediaFiles", () => ({
  storeMedia: jest.fn(async (conv: string, kind: string) => ({ path: `media/${conv}/x.${kind}` })),
  removeMedia: jest.fn(async () => {}),
  removeConversationMedia: jest.fn(async () => {}),
}));

const URL = "http://10.0.0.5:17434";

beforeEach(async () => {
  __resetChatStore();
  await AsyncStorage.clear();
  jest.clearAllMocks();
});

async function fresh(model = "sd") {
  const conv = newConversation(URL, model);
  await openConversation(conv.id, conv);
  return conv.id;
}

const imageStream = () =>
  response(fixture("images-stream.sse"), { contentType: "text/event-stream", chunk: 64 });

describe("historyFor", () => {
  it("leaves out generation prompts and their results", () => {
    const messages: Message[] = [
      { role: "user", content: "hello", at: 1 },
      { role: "assistant", content: "Hi!", at: 2 },
      { role: "user", content: "a fox", generate: true, at: 3 },
      {
        role: "assistant",
        content: "",
        at: 4,
        prompt: "a fox",
        request: { kind: "image" },
        media: { kind: "image", mime: "image/png", bytes: 10, path: "media/x.png" },
      },
      // A failed generation has no media, but is still not a chat turn.
      { role: "assistant", content: "", at: 5, request: { kind: "video" }, error: "no" },
      { role: "user", content: "and you?", at: 6 },
    ];
    expect(historyFor(messages, "Be brief.")).toEqual([
      { role: "system", content: "Be brief." },
      { role: "user", content: "hello" },
      { role: "assistant", content: "Hi!" },
      { role: "user", content: "and you?" },
    ]);
  });
});

it("generates an image: marks the prompt, reports steps, stores the result with the chat", async () => {
  const { fetch, calls } = fakeFetch(() => imageStream());
  const id = await fresh();
  setMediaOptions(id, { kind: "image", width: 64, height: 48, steps: 4, seed: 7 });
  const seen: (number | undefined)[] = [];
  const unsub = subscribe(() => seen.push(state(id).generation?.step));
  await send(new ApiClient({ url: URL }, fetch), id, "A red fox", {
    loaded: false,
    mediaKinds: ["image", "video"],
  });
  unsub();

  expect(JSON.parse(calls[0]!.init.body!)).toMatchObject({
    model: "sd",
    prompt: "A red fox",
    width: 64,
    steps: 4,
    seed: 7,
  });
  expect(seen).toEqual(expect.arrayContaining([undefined, 1, 2, 3, 4]));
  const { conv, phase, generation } = state(id);
  expect(phase).toBe("idle");
  expect(generation).toBeUndefined();
  expect(conv!.title).toBe("A red fox");
  expect(conv!.messages[0]).toMatchObject({ role: "user", content: "A red fox", generate: true });
  expect(conv!.messages[1]).toMatchObject({
    role: "assistant",
    content: "",
    model: "sd",
    prompt: "A red fox",
    request: { kind: "image", width: 64, height: 48, steps: 4, seed: 7 },
    media: {
      kind: "image",
      mime: "image/png",
      width: 64,
      height: 48,
      revisedPrompt: "a red fox in fresh snow, morning light",
      path: `media/${id}/x.image`,
    },
  });
  expect(storeMedia).toHaveBeenCalledWith(id, "image", expect.any(Uint8Array), "image/png");
  expect(await loadConversation(id)).toEqual(conv);
});

it("sends a later chat turn without the generation in its history", async () => {
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fakeFetch(() => imageStream()).fetch), id, "A fox", {
    mediaKinds: ["image"],
  });
  const chat = fakeFetch(() => response(fixture("chat-smollm.sse")));
  await send(new ApiClient({ url: URL }, chat.fetch), id, "Describe it", {});
  expect(JSON.parse(chat.calls[0]!.init.body!).messages).toEqual([{ role: "user", content: "Describe it" }]);
});

it("shows a failure as the reply's error", async () => {
  const { fetch } = fakeFetch(() =>
    response(fixture("images-error.sse"), { contentType: "text/event-stream" }),
  );
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fetch), id, "huge", { mediaKinds: ["image"] });
  const reply = state(id).conv!.messages[1]!;
  expect(reply.error).toBeTruthy();
  expect(reply.media).toBeUndefined();
  expect(storeMedia).not.toHaveBeenCalled();
});

it("drops a stopped image, keeping the prompt", async () => {
  const id = await fresh();
  const fetch = jest.fn(
    (_url: string, init?: { signal?: AbortSignal }) =>
      new Promise<ResponseLike>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const e = new Error("aborted");
          e.name = "AbortError";
          reject(e);
        });
      }),
  );
  const done = send(new ApiClient({ url: URL }, fetch), id, "a fox", { mediaKinds: ["image"] });
  await until(() => state(id).phase === "generating");
  stop(id);
  await done;
  expect(state(id).conv!.messages).toEqual([expect.objectContaining({ role: "user", content: "a fox" })]);
});

it("keeps a stopped video as a note: the daemon finishes it anyway", async () => {
  const id = await fresh("ltx");
  const fetch = jest.fn(
    (_url: string, init?: { signal?: AbortSignal }) =>
      new Promise<ResponseLike>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          const e = new Error("aborted");
          e.name = "AbortError";
          reject(e);
        });
      }),
  );
  const done = send(new ApiClient({ url: URL }, fetch), id, "waves", { mediaKinds: ["video"] });
  await until(() => state(id).phase === "generating");
  stop(id);
  await done;
  expect(state(id).conv!.messages[1]).toMatchObject({ stopped: true, request: { kind: "video" } });
});

it("retries a generation, deleting the replaced result's file", async () => {
  const id = await fresh();
  const client = new ApiClient({ url: URL }, fakeFetch(() => imageStream()).fetch);
  await send(client, id, "a fox", { mediaKinds: ["image"] });
  const first = state(id).conv!.messages[1]!.media;
  await retry(client, id, { mediaKinds: ["image"] });
  expect(removeMedia).toHaveBeenCalledWith(first);
  expect(state(id).conv!.messages).toHaveLength(2);
  expect(state(id).conv!.messages[1]!.media).toBeTruthy();
});

it("retries a generation as a chat turn after switching to a chat model", async () => {
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fakeFetch(() => imageStream()).fetch), id, "a fox", {
    mediaKinds: ["image"],
  });
  const chat = fakeFetch(() => response(fixture("chat-smollm.sse")));
  await retry(new ApiClient({ url: URL }, chat.fetch), id, {});
  const { messages } = state(id).conv!;
  expect(messages[0]!.generate).toBeUndefined();
  expect(messages[1]).toMatchObject({ content: "Hello!" });
  expect(JSON.parse(chat.calls[0]!.init.body!).messages).toEqual([{ role: "user", content: "a fox" }]);
});

it("exports without the media's bytes or file, keeping what it was", async () => {
  const id = await fresh();
  setMediaOptions(id, { kind: "image", steps: 4 });
  await send(new ApiClient({ url: URL }, fakeFetch(() => imageStream()).fetch), id, "a fox", {
    mediaKinds: ["image"],
  });
  const conv = state(id).conv!;
  const web = {
    ...conv,
    messages: conv.messages.map((m) =>
      m.media
        ? { ...m, media: { ...m.media, path: undefined, blob: new Blob(["png"], { type: "image/png" }) } }
        : m,
    ),
  };
  const out = JSON.parse(exportJson([conv, web]));
  // The chat's own options are not media: they stay.
  expect(out[0].media).toEqual({ kind: "image", steps: 4 });
  expect(out[0].messages[1].media).toMatchObject({ kind: "image", width: 64, omitted: true });
  expect(out[0].messages[1].media.path).toBeUndefined();
  expect(out[1].messages[1].media).toMatchObject({ kind: "image", omitted: true });
  expect(out[1].messages[1].media.blob).toBeUndefined();
});

async function until(ok: () => boolean) {
  for (let i = 0; i < 100 && !ok(); i++) await new Promise((r) => setImmediate(r));
  expect(ok()).toBe(true);
}
