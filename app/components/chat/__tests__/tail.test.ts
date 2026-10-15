import { tail } from "../MessageView";

it("keeps short text whole", () => {
  expect(tail("a\nb", 6)).toBe("a\nb");
});

it("keeps the last lines and marks the cut", () => {
  expect(tail("1\n2\n3\n4", 2)).toBe("…\n3\n4");
});

it("counts a long line by its wrapped rows and cuts it from the front", () => {
  expect(tail("x".repeat(30) + "END", 1, 10)).toBe("…" + "xxxxxxxEND");
});
