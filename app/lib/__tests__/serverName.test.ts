import { serverName, shortVersion } from "../serverName";

test.each([
  [{ url: "http://127.0.0.1:17434" }, "This computer"],
  [{ url: "http://localhost:17434" }, "This computer"],
  [{ url: "http://[::1]:17434" }, "This computer"],
  [{ url: "http://192.168.1.20:17434" }, "192.168.1.20"],
  [{ url: "https://studio.local" }, "studio.local"],
  [{ name: "Studio", url: "http://192.168.1.20:17434" }, "Studio"],
])("%j", (server, name) => {
  expect(serverName(server)).toBe(name);
});

test("the version drops its build", () => {
  expect(shortVersion("0.1.0 (5379f22-dirty)")).toBe("0.1.0");
  expect(shortVersion("0.2.1")).toBe("0.2.1");
});
