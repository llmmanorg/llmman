// A fake fetch that serves recorded daemon responses, split into chunks at
// awkward places (mid-line, mid-UTF-8 sequence) the way a socket would.

import { readFileSync } from "fs";
import { join } from "path";

import type { FetchLike, RequestInitLike, ResponseLike } from "@/api/client";

export function fixture(name: string): string {
  return readFileSync(join(__dirname, "..", "api", "__fixtures__", name), "utf8");
}

export interface Recorded {
  url: string;
  init: RequestInitLike;
}

export function chunked(text: string, size = 7): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= bytes.length) {
        controller.close();
        return;
      }
      controller.enqueue(bytes.slice(offset, offset + size));
      offset += size;
    },
  });
}

export function response(
  body: string,
  { status = 200, contentType = "application/json", chunk = 7, headers = {} as Record<string, string> } = {},
): ResponseLike {
  const all: Record<string, string> = { "content-type": contentType };
  for (const [k, v] of Object.entries(headers)) all[k.toLowerCase()] = v;
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "",
    headers: { get: (n: string) => all[n.toLowerCase()] ?? null },
    body: chunked(body, chunk),
    text: async () => body,
  };
}

/** A fetch answering every call with `reply(url, init)`, recording the calls. */
export function fakeFetch(
  reply: (url: string, init: RequestInitLike) => ResponseLike | Promise<ResponseLike>,
) {
  const calls: Recorded[] = [];
  const fetch: FetchLike = async (url, init = {}) => {
    calls.push({ url, init });
    return reply(url, init);
  };
  return { fetch, calls };
}

/** The daemon's own 401, as llmman's auth layer sends it. */
export function llmman401(): ResponseLike {
  return response(fixture("unauthorized.json"), {
    status: 401,
    headers: { "www-authenticate": 'Bearer realm="llmman"' },
  });
}
