/** @jest-environment node */

import { ApiClient, ApiError } from "../client";
import { explain } from "../errors";
import { fakeFetch, fixture, llmman401, response } from "@/test/fakeFetch";

const server = { name: "studio", url: "http://10.0.0.77:17435" };

it("says llmman's own 401 is a rejected key, not something to retry", async () => {
  const { fetch } = fakeFetch(() => llmman401());
  const e = await new ApiClient({ url: server.url, key: "old" }, fetch).listLocal().catch((x) => x);
  const out = explain(e, server);
  expect(out).toMatchObject({
    keyRejected: true,
    title: "studio rejected this phone's API key. Check it against LLMMAN_API_KEYS on the computer.",
  });
  // The daemon's CLI advice (header names, LLMMAN_API_KEY) stays out of the headline.
  expect(out.title).not.toMatch(/LLMMAN_API_KEY\b|x-api-key|Authorization/);
  expect(out.detail).toMatch(/^this llmman serve requires an API key/);
});

it("does not blame the key for a 401 that is not llmman's (a provider's, a proxy's)", () => {
  const out = explain(new ApiError('no API key for provider "anthropic"', 401, "http", false), server);
  expect(out).toEqual({ title: 'no API key for provider "anthropic"', keyRejected: false });
});

it("puts the connection's internals under a plain headline", () => {
  const raw =
    "could not reach http://10.0.0.77:17435: fetch failed: UnexpectedException: The network connection was lost. (at ExpoModulesCore/Promise.swift:56)";
  const out = explain(new ApiError(raw, 0, "network"), server);
  expect(out.title).toBe(
    "studio isn't answering. Is llmman serve running there, and is this phone on the same network?",
  );
  expect(out.title).not.toMatch(/swift|Exception/i);
  expect(out.detail).toBe(raw);
});

it("names an unnamed server by host and port", () => {
  expect(explain(new ApiError("x", 0, "timeout"), { url: server.url }).title).toBe(
    "10.0.0.77:17435 didn't answer in time. Is this phone on the same network?",
  );
});

it("keeps the daemon's own words for its other errors", async () => {
  const { fetch } = fakeFetch(() => response(fixture("chat-error-500.json"), { status: 500 }));
  const e = await new ApiClient({ url: server.url }, fetch)
    .chat({ model: "m", messages: [{ role: "user", content: "x" }] })
    .catch((x) => x);
  expect(explain(e, server)).toEqual({ title: expect.stringMatching(/^pull failed/), keyRejected: false });
  expect(explain(new Error("plain"), server).title).toBe("plain");
});
