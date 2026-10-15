import { groupChats, groupOf } from "@/lib/chatGroups";

const now = new Date(2026, 9, 7, 15, 0).getTime();
const at = (d: number, h = 12) => new Date(2026, 9, d, h, 0).getTime();

test("groups by local day", () => {
  expect(groupOf(at(7, 0), now)).toBe("Today");
  expect(groupOf(at(6, 23), now)).toBe("Yesterday");
  expect(groupOf(at(6, 0), now)).toBe("Yesterday");
  expect(groupOf(at(1), now)).toBe("Previous 7 days");
  expect(groupOf(at(29 - 30), now)).toBe("Older");
});

test("keeps order and drops empty groups", () => {
  const items = [at(7), at(7, 1), at(3), at(1 - 20)].map((updatedAt) => ({ updatedAt }));
  expect(groupChats(items, now).map((g) => [g.title, g.data.length])).toEqual([
    ["Today", 2],
    ["Previous 7 days", 1],
    ["Older", 1],
  ]);
});
