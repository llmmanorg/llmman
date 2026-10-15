// Generated media on a phone: files in the app's document directory, one
// folder per conversation, with their path (relative, since the app's
// container moves between installs and updates) kept in the chat. Saving
// one hands it to the share sheet, or to Photos. The web keeps Blobs in
// IndexedDB instead (mediaFiles.web.ts).

import { Directory, File, Paths } from "expo-file-system";
import { Asset, requestPermissionsAsync } from "expo-media-library";
import * as Sharing from "expo-sharing";
import { useMemo } from "react";

import { newId } from "@/lib/id";
import { extensionFor, type GeneratedMedia, type MediaKind } from "@/lib/media";

const ROOT = "media";

/** Keeps the bytes of a result for conversation `conv`; returns where they are. */
export async function storeMedia(
  conv: string,
  kind: MediaKind,
  bytes: Uint8Array,
  _mime: string,
): Promise<Pick<GeneratedMedia, "path" | "blob">> {
  const path = `${ROOT}/${conv}/${newId()}.${extensionFor(kind)}`;
  const file = new File(Paths.document, path);
  file.create({ intermediates: true, overwrite: true });
  file.write(bytes);
  return { path };
}

function fileOf(media: GeneratedMedia): File | null {
  return media.path ? new File(Paths.document, media.path) : null;
}

/** A URI the image, video or audio player can load, or null when the file is gone. */
export function useMediaUri(media: GeneratedMedia | undefined): string | null {
  return useMemo(() => {
    const file = media ? fileOf(media) : null;
    return file?.exists ? file.uri : null;
  }, [media]);
}

/** Deletes one result's file (a retried reply's). */
export async function removeMedia(media: GeneratedMedia | undefined): Promise<void> {
  try {
    const file = media ? fileOf(media) : null;
    if (file?.exists) file.delete();
  } catch {
    // already gone
  }
}

/** Deletes the media of deleted conversations. */
export async function removeConversationMedia(ids: string[]): Promise<void> {
  for (const id of ids) {
    try {
      const dir = new Directory(Paths.document, ROOT, id);
      if (dir.exists) dir.delete();
    } catch {
      // nothing there
    }
  }
}

/** A copy under the download name, so the share sheet and Photos show that name. */
function named(media: GeneratedMedia, name: string): File {
  const src = fileOf(media);
  if (!src?.exists) throw new Error("The file is no longer on this device.");
  const out = new File(Paths.cache, name);
  if (out.exists) out.delete();
  src.copy(out);
  return out;
}

/** The share sheet (Files, AirDrop, Drive…). */
export async function shareMedia(media: GeneratedMedia, name: string): Promise<void> {
  const file = named(media, name);
  await Sharing.shareAsync(file.uri, { mimeType: media.mime, dialogTitle: name });
}

/** Whether Photos can take this kind (pictures and videos, not sound). */
export function canSaveToPhotos(kind: MediaKind): boolean {
  return kind === "image" || kind === "video";
}

/** Adds a picture or video to the photo library, asking for permission to add (only). */
export async function saveToPhotos(media: GeneratedMedia, name: string): Promise<void> {
  const { granted } = await requestPermissionsAsync(true, [media.kind === "video" ? "video" : "photo"]);
  if (!granted) throw new Error("llmman may not add to your photo library. Allow it in Settings.");
  const file = named(media, name);
  await Asset.create(file.uri);
}

/** The web downloads instead (mediaFiles.web.ts). */
export function downloadMedia(_media: GeneratedMedia, _name: string): void {}
