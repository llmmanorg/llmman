import { popoverFrame } from "../pickerAnchor";

test("the popover sits under the title, centred, inside the window", () => {
  const f = popoverFrame({ x: 800, y: 14, width: 120, height: 36 }, { width: 1440, height: 900 }, 420);
  expect(f).toEqual({ left: 650, top: 56, width: 420, maxHeight: 832 });
});

test("it is kept inside a narrow window", () => {
  const f = popoverFrame({ x: 900, y: 10, width: 100, height: 36 }, { width: 1000, height: 700 }, 420);
  expect(f.left + f.width).toBeLessThanOrEqual(988);
});

test("without an anchor it is centred near the top", () => {
  expect(popoverFrame(null, { width: 1000, height: 800 }, 400).left).toBe(300);
});
