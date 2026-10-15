// Generated media in the browser: the Blob itself, stored with its
// conversation in IndexedDB (chatStorage.web.ts), shown through an object
// URL and saved as a download, as llmman's web UI does.

import { useEffect } from "react";

import type { GeneratedMedia, MediaKind } from "@/lib/media";

export async function storeMedia(
  _conv: string,
  _kind: MediaKind,
  bytes: Uint8Array,
  mime: string,
): Promise<Pick<GeneratedMedia, "path" | "blob">> {
  return { blob: new Blob([bytes as Uint8Array<ArrayBuffer>], { type: mime }) };
}

// Object URLs per Blob, shared by everything showing it and revoked a
// moment after the last one stops (a remount picks the same one up again).
const urls = new Map<Blob, { url: string; refs: number; timer?: ReturnType<typeof setTimeout> }>();

function entryFor(blob: Blob) {
  let e = urls.get(blob);
  if (!e) {
    e = { url: URL.createObjectURL(blob), refs: 0 };
    urls.set(blob, e);
  }
  return e;
}

/** An object URL for the media's Blob while the component shows it. */
export function useMediaUri(media: GeneratedMedia | undefined): string | null {
  const blob = media?.blob instanceof Blob ? media.blob : null;
  useEffect(() => {
    if (!blob) return;
    const e = entryFor(blob);
    e.refs++;
    if (e.timer) clearTimeout(e.timer);
    return () => {
      e.refs--;
      if (e.refs > 0) return;
      e.timer = setTimeout(() => {
        if (e.refs > 0) return;
        URL.revokeObjectURL(e.url);
        urls.delete(blob);
      }, 1000);
    };
  }, [blob]);
  return blob ? entryFor(blob).url : null;
}

/** The Blob goes with its conversation's record; nothing else to delete. */
export async function removeMedia(_media: GeneratedMedia | undefined): Promise<void> {}

export async function removeConversationMedia(_ids: string[]): Promise<void> {}

export function downloadMedia(media: GeneratedMedia, name: string): void {
  if (!(media.blob instanceof Blob)) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(media.blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export async function shareMedia(media: GeneratedMedia, name: string): Promise<void> {
  downloadMedia(media, name);
}

export function canSaveToPhotos(_kind: MediaKind): boolean {
  return false;
}

export async function saveToPhotos(media: GeneratedMedia, name: string): Promise<void> {
  downloadMedia(media, name);
}
