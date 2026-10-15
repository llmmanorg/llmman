Responses recorded from a real `llmman serve` (0.1.0) with curl, local paths
replaced by `/store/...`:

- `chat-smollm.sse`: `POST /v1/chat/completions` stream, SmolLM2-135M, ends `stop` + `[DONE]`.
- `chat-qwen-reasoning.sse`: same with qwen3.5 0.8b: `reasoning_content` deltas, cut by `max_tokens` (`length`).
- `pull-existing.ndjson`: `POST /api/pull` of a model already in the store.
- `pull-error.ndjson`: `POST /api/pull` of a repo that does not exist (`{error}` line in a 200 stream).
- `unauthorized.json`: the 401 body for a wrong key.
- `chat-error-500.json`: the 500 body for a chat with an unknown model.

Tests derive the truncated and progress cases from these.

## Models tab

Recorded 2026-10-07 from `llmman serve` (keyed, read-only GETs with curl), byte for byte:

- `search-popular.json`: `GET /llmman/search/popular?limit=20`; 17 Docker Hub rows, then 20 Hugging Face.
- `search-qwen.json`, `search-empty.json`: `GET /llmman/search?q=…&limit=…`, with hits and with none.
- `model-docker-qwen3.json`: `GET /llmman/search/model?name=docker.io/ai/qwen3`; its README opens with indented HTML.
- `model-docker-qwen3.5.json`: `GET /llmman/search/model?name=docker.io/ai/qwen3.5`; 128 variants, not sorted by size, `latest` the default.
- `model-hf-qwen3-coder-30b.json`, `model-hf-smollm2.json`, `model-hf-novariants.json`: Hugging Face GGUF repos; quantizations sorted by size, `Q4_K_M` the default.
- `model-gated.json`: a gated repo (`gated: true`, `readme: null`).
- `model-notfound.json`: the `502` body for a repo the registry refuses.
- `node.json`, `tags.json`, `v1-models.json`: `GET /llmman/node`, `/api/tags`, `/v1/models`.

- `model-hf-quantfactory-smollm2.json`: a card whose README opens with a blank line before its front matter.
- `pull-hf-q2k.ndjson`: `POST /api/pull` of `hf.co/QuantFactory/SmolLM2-135M-Instruct-GGUF:Q2_K` (88 MB),
  recorded with curl on 2026-10-07 with the user's approval and deleted afterwards: `pulling manifest`, `pulling`,
  then one total for the whole model polled every 200 ms (no `digest`), then `success`. This daemon has no
  signature policy, so it sends no `{notice}` line; the tests add one inline where they need it.

## Media generation

No media generation model under ~150 MB exists to record from, so these are
written to the exact shapes of llmman's handlers on upstream main
(`src/mediagen/server.rs`: `images`, `videos`), with a real 64×48 PNG:

- `images-stream.sse`: `POST /v1/images/generations` with `stream: true`: four
  `image_generation.progress` events, then `image_generation.completed` with
  `b64_json` and `revised_prompt`.
- `images-error.sse`: a progress event, then `{"type":"error","error":{"message"}}`.
- `images-plain.json`: the non-streamed answer, `{created, data: [{b64_json, revised_prompt}], usage}`.
- `video-job.json`: the completed job `POST /v1/videos` answers with, `size`,
  `seconds` (a string), `fps`, `n_frames`, `has_audio` and its `content_url`.
