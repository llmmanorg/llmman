/** @jest-environment node */

import { checkServer } from "../connect";
import { fakeFetch, llmman401, response } from "@/test/fakeFetch";

describe("checkServer", () => {
  it("rejects a bad address without a request", async () => {
    const { fetch, calls } = fakeFetch(() => response("{}"));
    expect(await checkServer({ address: "ftp://x" }, { fetchImpl: fetch })).toMatchObject({
      ok: false,
      field: "address",
      problem: "invalid",
    });
    expect(calls).toHaveLength(0);
  });

  it("points a 401 at the key field", async () => {
    const { fetch } = fakeFetch(() => llmman401());
    expect(await checkServer({ address: "10.0.0.2:17434" }, { fetchImpl: fetch })).toMatchObject({
      ok: false,
      field: "key",
      problem: "unauthorized",
    });
  });

  it("sends the trimmed key and keeps it, or none", async () => {
    const { fetch, calls } = fakeFetch(() => response('{"version":"0.2.0"}'));
    expect(await checkServer({ address: "10.0.0.2", key: " k " }, { fetchImpl: fetch })).toEqual({
      ok: true,
      url: "http://10.0.0.2:17434",
      key: "k",
      version: "0.2.0",
    });
    expect(calls[0]!.init.headers).toEqual({ authorization: "Bearer k" });
    const r = await checkServer({ address: "h:1" }, { fetchImpl: fetch });
    expect(r.ok && r.key).toBeUndefined();
  });
});
