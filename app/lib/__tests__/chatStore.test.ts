/** @jest-environment node */

import AsyncStorage from "@react-native-async-storage/async-storage";

import { ApiClient } from "@/api/client";
import { fakeFetch, fixture, llmman401, response } from "@/test/fakeFetch";
import {
  __resetChatStore,
  cutAt,
  editFrom,
  getConversationState as state,
  historyFor,
  openConversation,
  retry,
  send,
  setEffort,
  setModel,
  setSettings,
  stop,
} from "../chatStore";
import { loadConversation, newConversation } from "../conversations";

jest.mock("expo-crypto", () => {
  let n = 0;
  return { randomUUID: () => `c-${++n}` };
});

const URL = "http://10.0.0.5:17434";

beforeEach(async () => {
  __resetChatStore();
  await AsyncStorage.clear();
});

async function fresh(model = "m") {
  const conv = newConversation(URL, model);
  await openConversation(conv.id, conv);
  return conv.id;
}

it("sends, streams the reply, titles and saves the conversation", async () => {
  const { fetch, calls } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fetch), id, "Say hello");
  const { conv, phase } = state(id);
  expect(phase).toBe("idle");
  expect(conv!.title).toBe("Say hello");
  expect(conv!.messages).toEqual([
    expect.objectContaining({ role: "user", content: "Say hello" }),
    expect.objectContaining({ role: "assistant", content: "Hello!", finishReason: "stop", model: "m" }),
  ]);
  expect(JSON.parse(calls[0]!.init.body!).messages).toEqual([{ role: "user", content: "Say hello" }]);
  expect(await loadConversation(id)).toEqual(conv);
});

it("sends the chat's thinking level, and forgets it with another model", async () => {
  const { fetch, calls } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const id = await fresh("llmman.provider/anthropic/claude-sonnet-5-5");
  setEffort(id, "xhigh");
  await send(new ApiClient({ url: URL }, fetch), id, "hi");
  expect(JSON.parse(calls[0]!.init.body!).reasoning_effort).toBe("xhigh");
  expect((await loadConversation(id))!.effort).toBe("xhigh");

  setModel(id, "m");
  expect(state(id).conv!.effort).toBeUndefined();
  await send(new ApiClient({ url: URL }, fetch), id, "again");
  expect(JSON.parse(calls[1]!.init.body!).reasoning_effort).toBeUndefined();
});

it("sends the chat's system prompt, temperature and max tokens", async () => {
  const { fetch, calls } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const conv = newConversation(URL, "m", { systemPrompt: "Be brief.", temperature: 0.2, maxTokens: 32 });
  await openConversation(conv.id, conv);
  await send(new ApiClient({ url: URL }, fetch), conv.id, "hi");
  expect(JSON.parse(calls[0]!.init.body!)).toEqual({
    model: "m",
    stream: true,
    temperature: 0.2,
    max_tokens: 32,
    messages: [
      { role: "system", content: "Be brief." },
      { role: "user", content: "hi" },
    ],
  });
});

it("changes a chat's settings for its next message and saves them", async () => {
  const { fetch, calls } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const client = new ApiClient({ url: URL }, fetch);
  const conv = newConversation(URL, "m", { temperature: 1 });
  await openConversation(conv.id, conv);
  await send(client, conv.id, "a");
  setSettings(conv.id, { maxTokens: 8 });
  await send(client, conv.id, "b");
  const second = JSON.parse(calls[1]!.init.body!);
  expect(second.max_tokens).toBe(8);
  expect(second).not.toHaveProperty("temperature");
  await new Promise((r) => setTimeout(r, 0));
  expect((await loadConversation(conv.id))!.settings).toEqual({ maxTokens: 8 });
  setSettings(conv.id, {});
  expect(state(conv.id).conv).not.toHaveProperty("settings");
});

it("keeps reasoning apart from the answer", async () => {
  const { fetch } = fakeFetch(() => response(fixture("chat-qwen-reasoning.sse"), { chunk: 64 }));
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fetch), id, "2+3?");
  const reply = state(id).conv!.messages[1]!;
  expect(reply.reasoning!.startsWith("Thinking Process:")).toBe(true);
  expect(reply.finishReason).toBe("length");
});

it("records a failure on the reply and keeps the prompt", async () => {
  const { fetch } = fakeFetch(() => response('{"error":"model not found"}', { status: 404 }));
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fetch), id, "hi");
  expect(state(id).conv!.messages[1]).toMatchObject({ role: "assistant", error: "model not found" });
});

it("words a rejected key and a dead connection the way the Models tab does", async () => {
  const { fetch } = fakeFetch(() => llmman401());
  const id = await fresh();
  await send(new ApiClient({ url: URL, key: "old" }, fetch), id, "hi", {
    server: { name: "studio", url: URL },
  });
  expect(state(id).conv!.messages[1]).toMatchObject({
    error: "studio rejected this phone's API key. Check it against LLMMAN_API_KEYS on the computer.",
    errorDetail: expect.stringMatching(/requires an API key/),
    keyRejected: true,
  });

  const down = async () => {
    throw new TypeError("UnexpectedException (at ExpoModulesCore/Promise.swift:56)");
  };
  const id2 = await fresh();
  await send(new ApiClient({ url: URL }, down), id2, "hi");
  const msg = state(id2).conv!.messages[1]!;
  expect(msg.error).toBe(
    "10.0.0.5:17434 isn't answering. Is llmman serve running there, and is this phone on the same network?",
  );
  expect(msg.errorDetail).toMatch(/Promise\.swift/);
  expect(msg).not.toHaveProperty("keyRejected");
});

it("drops an empty reply when stopped before any token", async () => {
  const fetch = (_u: string, init?: { signal?: AbortSignal }) =>
    new Promise<never>((_, reject) =>
      init?.signal?.addEventListener("abort", () => {
        const e = new Error("aborted");
        e.name = "AbortError";
        reject(e);
      }),
    );
  const id = await fresh();
  const p = send(new ApiClient({ url: URL }, fetch), id, "hi");
  await new Promise((r) => setTimeout(r, 10));
  expect(state(id).phase).toBe("thinking");
  stop(id);
  await p;
  expect(state(id).conv!.messages).toEqual([expect.objectContaining({ role: "user" })]);
  expect(state(id).phase).toBe("idle");
});

it("keeps a partial reply marked stopped", async () => {
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"part"}}]}\n\n'));
    },
  });
  const id = await fresh();
  const client = new ApiClient({ url: URL }, async () => ({ ...response(""), body }));
  const p = send(client, id, "hi");
  await new Promise((r) => setTimeout(r, 100));
  stop(id);
  await p;
  expect(state(id).conv!.messages[1]).toMatchObject({ content: "part", stopped: true });
});

it("says so when the model returns nothing", async () => {
  const { fetch } = fakeFetch(() => response("data: [DONE]\n\n"));
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fetch), id, "hi");
  expect(state(id).conv!.messages[1]).toMatchObject({ error: "The model returned nothing." });
});

it("retries by replacing the last reply, and sends only answered turns as history", async () => {
  let n = 0;
  const { fetch, calls } = fakeFetch(() =>
    n++ === 0 ? response('{"error":"busy"}', { status: 503 }) : response(fixture("chat-smollm.sse")),
  );
  const client = new ApiClient({ url: URL }, fetch);
  const id = await fresh();
  await send(client, id, "hi");
  await retry(client, id);
  expect(state(id).conv!.messages).toHaveLength(2);
  expect(state(id).conv!.messages[1]).toMatchObject({ content: "Hello!" });
  expect(JSON.parse(calls[1]!.init.body!).messages).toEqual([{ role: "user", content: "hi" }]);
});

it("retries an earlier reply by cutting the conversation after it", async () => {
  const { fetch, calls } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const client = new ApiClient({ url: URL }, fetch);
  const id = await fresh();
  await send(client, id, "first");
  await send(client, id, "second");
  expect(state(id).conv!.messages).toHaveLength(4);
  await retry(client, id, {}, 1);
  const messages = state(id).conv!.messages;
  expect(messages.map((m) => [m.role, m.content])).toEqual([
    ["user", "first"],
    ["assistant", "Hello!"],
  ]);
  // Only what came before the retried reply is sent.
  expect(JSON.parse(calls[2]!.init.body!).messages).toEqual([{ role: "user", content: "first" }]);
  expect((await loadConversation(id))!.messages).toHaveLength(2);
});

it("times a reply", async () => {
  const { fetch } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fetch), id, "hi");
  expect(state(id).conv!.messages[1]!.durationMs).toEqual(expect.any(Number));
});

it("edits a user message: the conversation is cut there and its text comes back", async () => {
  const { fetch } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const client = new ApiClient({ url: URL }, fetch);
  const id = await fresh();
  await send(client, id, "first");
  await send(client, id, "second");
  expect(await editFrom(id, 2)).toEqual({ text: "second", images: [] });
  expect(state(id).conv!.messages.map((m) => m.content)).toEqual(["first", "Hello!"]);
  expect((await loadConversation(id))!.messages).toHaveLength(2);
  // Not a reply, and not past the end.
  expect(await editFrom(id, 1)).toBeNull();
  expect(await editFrom(id, 9)).toBeNull();
});

it("editing the first message leaves an empty chat, removed from storage", async () => {
  const { fetch } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const id = await fresh();
  await send(new ApiClient({ url: URL }, fetch), id, "only");
  expect(await editFrom(id, 0)).toEqual({ text: "only", images: [] });
  expect(state(id).conv!.messages).toEqual([]);
  expect(await loadConversation(id)).toBeNull();
});

it("cutAt keeps what is before the index", () => {
  const m = [0, 1, 2].map((i) => ({ role: "user" as const, content: String(i), at: i }));
  expect(cutAt(m, 1)).toEqual({ kept: [m[0]], dropped: [m[1], m[2]] });
  expect(cutAt(m, 0).kept).toEqual([]);
});

it("ignores a second send while one is running", async () => {
  const { fetch, calls } = fakeFetch(() => response(fixture("chat-smollm.sse")));
  const client = new ApiClient({ url: URL }, fetch);
  const id = await fresh();
  await Promise.all([send(client, id, "a"), send(client, id, "b")]);
  expect(calls).toHaveLength(1);
});

it("historyFor skips failed and empty replies", () => {
  expect(
    historyFor([
      { role: "user", content: "a", at: 1 },
      { role: "assistant", content: "", error: "x", at: 2 },
      { role: "user", content: "b", at: 3 },
      { role: "assistant", content: "ok", at: 4 },
    ]),
  ).toEqual([
    { role: "user", content: "a" },
    { role: "user", content: "b" },
    { role: "assistant", content: "ok" },
  ]);
});
