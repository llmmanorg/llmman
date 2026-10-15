// Where the chosen model sits in the model picker's sections, so the
// picker can open scrolled to it (and to its effort levels under it).

type PickerRow = { kind: string; id?: string; ref?: string };

/** The section and item of the row for `current`, or null when it is not listed (yet). */
export function selectedLocation(
  sections: { data: PickerRow[] }[],
  current: string | undefined,
): { sectionIndex: number; itemIndex: number } | null {
  if (!current) return null;
  for (let s = 0; s < sections.length; s++) {
    const i = sections[s]!.data.findIndex((r) =>
      r.kind === "hosted"
        ? r.ref === current
        : (r.kind === "local" || r.kind === "media") && r.id === current,
    );
    if (i >= 0) return { sectionIndex: s, itemIndex: i };
  }
  return null;
}

/** The models a keyboard can move through, in order, with where each row is. */
export function choosable(
  sections: { data: PickerRow[] }[],
): { ref: string; sectionIndex: number; itemIndex: number }[] {
  const out: { ref: string; sectionIndex: number; itemIndex: number }[] = [];
  sections.forEach((s, sectionIndex) =>
    s.data.forEach((r, itemIndex) => {
      const ref = r.kind === "hosted" ? r.ref : r.kind === "local" || r.kind === "media" ? r.id : undefined;
      if (ref) out.push({ ref, sectionIndex, itemIndex });
    }),
  );
  return out;
}

/** The next model for an arrow key: down from nothing is the first, and it stops at the ends. */
export function step(refs: string[], current: string | null, delta: 1 | -1): string | null {
  if (!refs.length) return null;
  const i = current === null ? -1 : refs.indexOf(current);
  if (i < 0) return delta === 1 ? refs[0]! : refs[refs.length - 1]!;
  return refs[Math.max(0, Math.min(refs.length - 1, i + delta))]!;
}
