import { choosable, selectedLocation, step } from "@/lib/pickerLocation";

const sections = [
  {
    data: [
      { kind: "local", id: "a" },
      { kind: "local", id: "b" },
    ],
  },
  {
    data: [
      { kind: "provider", id: "anthropic" },
      { kind: "hosted", ref: "llmman.provider/anthropic/x" },
      { kind: "hosted", ref: "llmman.provider/anthropic/y" },
    ],
  },
];

test("finds a local or a hosted model's row", () => {
  expect(selectedLocation(sections, "b")).toEqual({ sectionIndex: 0, itemIndex: 1 });
  expect(selectedLocation(sections, "llmman.provider/anthropic/y")).toEqual({
    sectionIndex: 1,
    itemIndex: 2,
  });
});

test("nothing for no model, or one not listed yet (a provider still closed)", () => {
  expect(selectedLocation(sections, undefined)).toBeNull();
  expect(selectedLocation(sections, "anthropic")).toBeNull();
  expect(selectedLocation(sections, "llmman.provider/openai/z")).toBeNull();
});

test("finds a media generation model's row under Generate", () => {
  const withMedia = [sections[0]!, { data: [{ kind: "media", id: "sd-turbo" }] }, sections[1]!];
  expect(selectedLocation(withMedia, "sd-turbo")).toEqual({ sectionIndex: 1, itemIndex: 0 });
});

test("the keyboard moves through models, not provider rows", () => {
  const refs = choosable(sections).map((r) => r.ref);
  expect(refs).toEqual(["a", "b", "llmman.provider/anthropic/x", "llmman.provider/anthropic/y"]);
  expect(step(refs, null, 1)).toBe("a");
  expect(step(refs, null, -1)).toBe("llmman.provider/anthropic/y");
  expect(step(refs, "b", 1)).toBe("llmman.provider/anthropic/x");
  expect(step(refs, "a", -1)).toBe("a");
  expect(step([], null, 1)).toBeNull();
});
