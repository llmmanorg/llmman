import { fitWithin } from "@/lib/attachments";
import { historyFor } from "@/lib/chatStore";

jest.mock("expo-image-manipulator", () => ({ ImageManipulator: {}, SaveFormat: { JPEG: "jpeg" } }));
jest.mock("expo-image-picker", () => ({}));

test("shrinks only what is larger than the long edge, keeping the aspect", () => {
  expect(fitWithin(1000, 800)).toBeNull();
  expect(fitWithin(1568, 1568)).toBeNull();
  expect(fitWithin(4032, 3024)).toEqual({ width: 1568 });
  expect(fitWithin(3024, 4032)).toEqual({ height: 1568 });
});

test("a message's images are sent as image_url parts after its text", () => {
  const url = "data:image/jpeg;base64,AAA";
  expect(
    historyFor([
      { role: "user", content: "What is this?", images: [url], at: 1 },
      { role: "assistant", content: "A cat.", at: 2 },
      { role: "user", content: "", images: [url], at: 3 },
    ]),
  ).toEqual([
    {
      role: "user",
      content: [
        { type: "text", text: "What is this?" },
        { type: "image_url", image_url: { url } },
      ],
    },
    { role: "assistant", content: "A cat." },
    { role: "user", content: [{ type: "image_url", image_url: { url } }] },
  ]);
});
