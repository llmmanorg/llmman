// Where the header's model title is on screen, so the wide-screen model
// menu opens as a popover right under it.

export interface Anchor {
  x: number;
  y: number;
  width: number;
  height: number;
}

let anchor: Anchor | null = null;

export function setPickerAnchor(a: Anchor | null): void {
  anchor = a;
}

export function pickerAnchor(): Anchor | null {
  return anchor;
}

/**
 * The popover's place: under the anchor, centred on it, kept inside the
 * window with `margin` on each side; without an anchor, centred near the top.
 */
export function popoverFrame(
  a: Anchor | null,
  window: { width: number; height: number },
  width: number,
  margin = 12,
): { left: number; top: number; width: number; maxHeight: number } {
  const w = Math.min(width, window.width - 2 * margin);
  const top = a ? a.y + a.height + 6 : 72;
  const centre = a ? a.x + a.width / 2 : window.width / 2;
  const left = Math.max(margin, Math.min(window.width - margin - w, centre - w / 2));
  return { left, top, width: w, maxHeight: Math.max(200, window.height - top - margin) };
}
