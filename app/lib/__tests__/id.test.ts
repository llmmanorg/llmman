import { newId, uuidFrom } from "@/lib/id";

jest.mock("expo-crypto", () => ({
  randomUUID: () => {
    throw new TypeError("crypto.randomUUID is not a function");
  },
}));

test("a v4 UUID from bytes", () => {
  expect(uuidFrom(new Uint8Array(16).fill(0xff))).toBe("ffffffff-ffff-4fff-bfff-ffffffffffff");
  expect(uuidFrom(new Uint8Array(16))).toBe("00000000-0000-4000-8000-000000000000");
});

test("falls back to getRandomValues where randomUUID is missing (an http:// page on the LAN)", () => {
  const a = newId();
  expect(a).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  expect(newId()).not.toBe(a);
});
