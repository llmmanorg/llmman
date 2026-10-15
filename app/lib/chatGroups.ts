// The drawer's conversation list, grouped by when each chat was last used.

export type ChatGroup = "Today" | "Yesterday" | "Previous 7 days" | "Older";

const DAY = 24 * 60 * 60 * 1000;

/** The group for a chat last used at `at`, seen at `now` (local days). */
export function groupOf(at: number, now = Date.now()): ChatGroup {
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);
  const start = today.getTime();
  if (at >= start) return "Today";
  if (at >= start - DAY) return "Yesterday";
  if (at >= start - 7 * DAY) return "Previous 7 days";
  return "Older";
}

/** Items (newest first) in their groups, empty groups left out. */
export function groupChats<T extends { updatedAt: number }>(
  items: T[],
  now = Date.now(),
): { title: ChatGroup; data: T[] }[] {
  const out: { title: ChatGroup; data: T[] }[] = [];
  for (const item of items) {
    const title = groupOf(item.updatedAt, now);
    const last = out[out.length - 1];
    if (last?.title === title) last.data.push(item);
    else out.push({ title, data: [item] });
  }
  return out;
}
