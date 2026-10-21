import { documentTitle, hashRoute, isSheet } from "@/lib/webRoutes";

test("llmman's web UI fragment links map to the app's routes", () => {
  expect(hashRoute("#/models")).toBe("/models");
  expect(hashRoute("#/shell")).toBe("/shell");
  expect(hashRoute("#/new")).toBe("/");
  expect(hashRoute("#/c/m1x2y3")).toBe("/c/m1x2y3");
});

test("anything else stays where it is", () => {
  expect(hashRoute("")).toBeNull();
  expect(hashRoute("#")).toBeNull();
  expect(hashRoute("#/")).toBeNull();
  expect(hashRoute("#/c/")).toBeNull();
  expect(hashRoute("#/c/a/b")).toBeNull();
  expect(hashRoute("#section-2")).toBeNull();
  expect(hashRoute("#/settings")).toBeNull();
});

test("the page is named per screen, as the web UI names its views", () => {
  expect(documentTitle("/")).toBe("llmman");
  expect(documentTitle("/c/abc")).toBe("llmman");
  expect(documentTitle("/shell")).toBe("Shell · llmman");
  expect(documentTitle("/models")).toBe("Models · llmman");
  expect(documentTitle("/models/hf.co%2Fx")).toBe("Models · llmman");
  expect(documentTitle("/settings")).toBe("Settings · llmman");
});

test("Escape closes sheets only", () => {
  expect(isSheet("/model-picker")).toBe(true);
  expect(isSheet("/chat-settings")).toBe(true);
  expect(isSheet("/generate-settings")).toBe(true);
  expect(isSheet("/")).toBe(false);
  expect(isSheet("/models")).toBe(false);
});
