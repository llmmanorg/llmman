// Images attached to a chat message: picked (library, camera, or on the web
// a file, a paste or a drop), shrunk on the device to at most 1568 px on the
// long edge as JPEG ~0.85, and sent as `data:` URLs (OpenAI `image_url`).

import { ImageManipulator, SaveFormat } from "expo-image-manipulator";
import * as ImagePicker from "expo-image-picker";

/** The long edge models take without scaling (Anthropic's guidance; OpenAI's high detail is similar). */
export const MAX_EDGE = 1568;
export const JPEG_QUALITY = 0.85;
/** At most this many images per message. */
export const MAX_IMAGES = 4;

/** The resize that fits `width`×`height` within `max` on the long edge, or null when it already fits. */
export function fitWithin(
  width: number,
  height: number,
  max = MAX_EDGE,
): { width: number } | { height: number } | null {
  if (width <= max && height <= max) return null;
  return width >= height ? { width: max } : { height: max };
}

/**
 * The image at `uri` as a `data:` URL, shrunk to fit: JPEG, except a PNG
 * stays PNG (JPEG has no transparency, so a transparent PNG would turn black).
 */
export async function prepareImage(uri: string, mimeType?: string | null): Promise<string> {
  const png = mimeType === "image/png";
  const original = await ImageManipulator.manipulate(uri).renderAsync();
  const fit = fitWithin(original.width, original.height);
  const image = fit ? await ImageManipulator.manipulate(uri).resize(fit).renderAsync() : original;
  const saved = await image.saveAsync(
    png
      ? { format: SaveFormat.PNG, base64: true }
      : { format: SaveFormat.JPEG, compress: JPEG_QUALITY, base64: true },
  );
  if (!saved.base64) throw new Error("The image could not be read.");
  return `data:image/${png ? "png" : "jpeg"};base64,${saved.base64}`;
}

/** Images chosen from the library (several) or taken with the camera; empty when cancelled. */
export async function pickImages(source: "library" | "camera", limit = MAX_IMAGES): Promise<string[]> {
  if (source === "camera") {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) throw new Error("llmman has no access to the camera. Allow it in Settings.");
  }
  const options: ImagePicker.ImagePickerOptions = {
    mediaTypes: ["images"],
    allowsMultipleSelection: source === "library" && limit > 1,
    selectionLimit: limit,
    quality: 1,
  };
  const result =
    source === "camera"
      ? await ImagePicker.launchCameraAsync(options)
      : await ImagePicker.launchImageLibraryAsync(options);
  if (result.canceled) return [];
  return Promise.all(result.assets.slice(0, limit).map((a) => prepareImage(a.uri, a.mimeType)));
}
