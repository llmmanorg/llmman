// The drawer's conversation list, grouped by when each chat was last used.

export type ChatGroup = "Today" | "Yesterday" | "Previous 7 days" | "Older";

/** Local midnight `days` days before `now`'s: a local day may be 23 or 25 hours long. */
function midnight(now: number, days: number): number {
  const d = new Date(now);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - days);
  return d.getTime();
}

/** The group for a chat last used at `at`, seen at `now` (local days). */
export function groupOf(at: number, now = Date.now()): ChatGroup {
  if (at >= midnight(now, 0)) return "Today";
  if (at >= midnight(now, 1)) return "Yesterday";
  if (at >= midnight(now, 7)) return "Previous 7 days";
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
