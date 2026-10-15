/** @jest-environment node */

import { ApiClient } from "@/api/client";
import { fakeFetch, response } from "@/test/fakeFetch";
import { fetchModels, generative } from "../ModelsContext";

const URL = "http://10.0.0.5:17434";

const caps: Record<string, string[] | null> = {
  "docker.io/ai/smollm2": ["completion"],
  "hf.co/x/sd-turbo": ["image"],
  "hf.co/x/ltx-video": ["video", "image", "audio"],
  "hf.co/x/embed": ["embedding"],
  "hf.co/x/flaky": null,
};

function daemon() {
  return fakeFetch((url, init) => {
    if (url.endsWith("/v1/models")) {
      return response(
        JSON.stringify({
          data: Object.keys(caps).map((id) => ({
            id,
            status: { value: id.includes("sd") ? "loaded" : "unloaded" },
          })),
        }),
      );
    }
    if (url.endsWith("/llmman/providers")) return response('{"providers":[]}');
    if (url.endsWith("/api/show")) {
      const { model } = JSON.parse(init.body!);
      if (caps[model] === null) return response('{"error":"boom"}', { status: 500 });
      return response(JSON.stringify({ capabilities: caps[model] }));
    }
    return response("{}", { status: 404 });
  });
}

it("lists chat models and media generation models with what they make, as the web UI does", async () => {
  const { fetch } = daemon();
  const { local } = await fetchModels(new ApiClient({ url: URL }, fetch), new Map());
  expect(local.map((m) => [m.id, m.kinds, m.loaded])).toEqual([
    ["docker.io/ai/smollm2", [], false],
    // A failed /api/show counts as chattable rather than hiding the model.
    ["hf.co/x/flaky", [], false],
    ["hf.co/x/ltx-video", ["image", "video", "audio"], false],
    ["hf.co/x/sd-turbo", ["image"], true],
  ]);
});

it("tells a media model from a chat model that also has an image capability", () => {
  expect(generative({ capabilities: ["image"] })).toBe(true);
  expect(generative({ capabilities: ["completion", "image"] })).toBe(false);
  expect(generative({ capabilities: ["embedding"] })).toBe(false);
  expect(generative({ capabilities: null })).toBe(false);
});
