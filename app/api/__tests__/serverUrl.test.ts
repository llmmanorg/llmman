import { DEFAULT_PORT, hostLabel, normalizeServerUrl } from "../serverUrl";
describe("normalizeServerUrl", () => {
  it.each([
    ["192.168.1.20", `http://192.168.1.20:${DEFAULT_PORT}`],
    ["192.168.1.20:17435", "http://192.168.1.20:17435"],
    ["  Studio.Local:8080/  ", "http://studio.local:8080"],
    ["http://10.0.0.77:17435", "http://10.0.0.77:17435"],
    ["HTTP://10.0.0.77", "http://10.0.0.77"],
    ["https://llm.example.com", "https://llm.example.com"],
    ["https://gw.example.com/llmman/", "https://gw.example.com/llmman"],
    ["http://[fe80::1]:17434", "http://[fe80::1]:17434"],
    ["[::1]", `http://[::1]:${DEFAULT_PORT}`],
    ["localhost:08080", "http://localhost:8080"],
  ])("%s → %s", (input, expected) => {
    expect(normalizeServerUrl(input)).toEqual({ ok: true, value: expected });
  });

  it.each([
    ["", /Enter/],
    ["   ", /Enter/],
    ["ftp://host", /http:\/\/ or https:\/\//],
    ["ws://host:1", /http:\/\/ or https:\/\//],
    ["host:70000", /port/],
    ["host:0", /port/],
    ["http://host?x=1", /\? and #/],
    ["http://user@host", /doesn't look like an address/],
    ["two words", /doesn't look like an address/],
  ])("rejects %j", (input, message) => {
    const r = normalizeServerUrl(input);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(message);
  });
});

it("hostLabel drops the scheme", () => {
  expect(hostLabel("http://10.0.0.77:17435")).toBe("10.0.0.77:17435");
});
