import { ApiClient, type KeySource } from "@/api/client";
import { createKeyPrompt } from "@/lib/keyPrompt";
import { fakeFetch, llmman401, response } from "@/test/fakeFetch";

const URL = "http://127.0.0.1:17492";

/** A page's key store and prompt, with the answers the user gives in turn. */
function page(answers: (string | null)[], initial = "") {
  let key = initial;
  const asked = jest.fn(async () => answers.shift() ?? null);
  const prompt = createKeyPrompt(asked, (k) => {
    key = k;
  });
  const source: KeySource = { get: () => key, ask: () => prompt.ask() };
  return { source, asked, prompt, key: () => key };
}

/** A daemon with key `good`: llmman's 401 for anything else. */
function keyedDaemon() {
  return fakeFetch((_url, init) =>
    init.headers?.authorization === "Bearer good" ? response('{"version":"0.3.0"}') : llmman401(),
  );
}

test("llmman's 401 asks for the key and retries the request once with it", async () => {
  const p = page(["good"]);
  const { fetch, calls } = keyedDaemon();
  const client = new ApiClient({ url: URL }, fetch, { keySource: p.source });
  expect((await client.version()).version).toBe("0.3.0");
  expect(p.asked).toHaveBeenCalledTimes(1);
  expect(calls.map((c) => c.init.headers?.authorization)).toEqual([undefined, "Bearer good"]);
  expect(p.key()).toBe("good");
});

test("any request's 401 asks, not only the first one's", async () => {
  // The key worked at first, then was rotated on the computer.
  const p = page(["good"], "rotated");
  const { fetch, calls } = fakeFetch((url, init) =>
    init.headers?.authorization === "Bearer good"
      ? response('{"data":[{"id":"m","status":{"value":"loaded"}}]}')
      : llmman401(),
  );
  const client = new ApiClient({ url: URL }, fetch, { keySource: p.source });
  expect(await client.listLocal()).toEqual([{ id: "m", loaded: true }]);
  expect(p.asked).toHaveBeenCalledTimes(1);
  expect(calls.map((c) => c.init.headers?.authorization)).toEqual(["Bearer rotated", "Bearer good"]);
});

test("concurrent 401s share one prompt, and each retries once", async () => {
  const p = page(["good"]);
  const { fetch, calls } = keyedDaemon();
  const client = new ApiClient({ url: URL }, fetch, { keySource: p.source });
  const all = await Promise.all([client.version(), client.version(), client.version()]);
  expect(all.map((v) => v.version)).toEqual(["0.3.0", "0.3.0", "0.3.0"]);
  expect(p.asked).toHaveBeenCalledTimes(1);
  expect(calls).toHaveLength(6);
});

test("a wrong key is retried once, then the 401 stands", async () => {
  const p = page(["bad"]);
  const { fetch, calls } = keyedDaemon();
  const client = new ApiClient({ url: URL }, fetch, { keySource: p.source });
  await expect(client.version()).rejects.toMatchObject({ status: 401, unauthorized: true });
  expect(calls).toHaveLength(2);
});

test("a cancelled prompt is not retried, nor shown again until reset", async () => {
  const p = page([null, "good"]);
  const { fetch, calls } = keyedDaemon();
  const client = new ApiClient({ url: URL }, fetch, { keySource: p.source });
  await expect(client.version()).rejects.toMatchObject({ status: 401 });
  await expect(client.version()).rejects.toMatchObject({ status: 401 });
  expect(p.asked).toHaveBeenCalledTimes(1);
  expect(calls).toHaveLength(2);

  p.prompt.reset();
  expect((await client.version()).version).toBe("0.3.0");
  expect(p.asked).toHaveBeenCalledTimes(2);
});

test("an empty answer counts as cancelled", async () => {
  const p = page(["   "]);
  expect(await p.prompt.ask()).toBe(false);
  expect(p.key()).toBe("");
});

test("a provider's 401 (no llmman realm) is an error, never a prompt", async () => {
  const p = page(["good"]);
  const { fetch, calls } = fakeFetch(() =>
    response('{"error":{"message":"invalid x-api-key"}}', { status: 401 }),
  );
  const client = new ApiClient({ url: URL }, fetch, { keySource: p.source });
  await expect(client.version()).rejects.toMatchObject({ status: 401, unauthorized: false });
  expect(p.asked).not.toHaveBeenCalled();
  expect(calls).toHaveLength(1);
});

test("without a key source a 401 is just an error (a phone keeps its key per server)", async () => {
  const { fetch, calls } = keyedDaemon();
  await expect(new ApiClient({ url: URL, key: "bad" }, fetch).version()).rejects.toMatchObject({
    status: 401,
  });
  expect(calls).toHaveLength(1);
});
