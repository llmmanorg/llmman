// Pictures in a prompt: shrinking them, choosing which go with the next
// request, and the camera. A picture is a JPEG Blob on its message and
// becomes base64 only in the request.

import { $ } from "./util.js";

export const MAX_IMAGES = 4;

/**
 * Bytes of one chat request. The daemon reads at most 2 MiB of it (axum's
 * default; only transcription lifts that), so the text and the pictures
 * share this, with room for JSON.
 */
export const REQUEST_BUDGET = 1_900_000;

/** A picture's size once shrunk; `MAX_IMAGES` of them (as base64) leave room for text. */
const IMAGE_BYTES = 300_000;

/** [longest side, JPEG quality], tried in turn until the picture is small enough. */
const STEPS = [
  [1280, 0.85],
  [1280, 0.7],
  [1024, 0.7],
  [768, 0.65],
  [512, 0.6],
];

/** What a picture adds to a request: its base64, the `data:` prefix and the JSON around it. */
export const requestSize = (bytes) => Math.ceil(bytes / 3) * 4 + 100;

/** Any image the browser decodes, as `{blob, width, height}`, a JPEG. */
export async function prepare(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error(`${file.name || "That file"} is not an image this browser can read`);
  }
  try {
    let best = null;
    for (const [side, quality] of STEPS) {
      const scale = Math.min(1, side / Math.max(bitmap.width, bitmap.height));
      const width = Math.max(1, Math.round(bitmap.width * scale));
      const height = Math.max(1, Math.round(bitmap.height * scale));
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff"; // JPEG has no alpha: a transparent PNG would be black
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(bitmap, 0, 0, width, height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
      if (!blob) throw new Error("This browser could not encode the image");
      best = { blob, width, height };
      if (blob.size <= IMAGE_BYTES) break;
    }
    return best;
  } finally {
    bitmap.close?.();
  }
}

/** A Blob as a `data:` URL, the form a chat request carries an image in. */
export function toDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/**
 * The pictures of `messages` that go with the next request: the newest
 * that fit `budget` bytes. The rest stay in the conversation; the model
 * is just not shown them again.
 */
export function imagesToSend(messages, budget) {
  const keep = new Set();
  for (const message of messages.toReversed()) {
    for (const image of (message.images ?? []).toReversed()) {
      const cost = requestSize(image.blob.size);
      if (cost > budget) continue;
      budget -= cost;
      keep.add(image);
    }
  }
  return keep;
}

/** A chat message's `content`: the text, then the pictures as `image_url` parts if any. */
export function contentParts(text, dataUrls) {
  if (!dataUrls.length) return text;
  return [{ type: "text", text }, ...dataUrls.map((url) => ({ type: "image_url", image_url: { url } }))];
}

// ---- Camera ---------------------------------------------------------------

/** Whether the page can open a camera itself (it needs a secure context). */
export function canCapture() {
  return !!navigator.mediaDevices?.getUserMedia;
}

/** Why a camera or microphone could not be opened, for a toast. */
export function deviceError(e, what = "camera") {
  switch (e?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return `Allow ${what} access to use it`;
    case "NotFoundError":
    case "OverconstrainedError":
      return `No ${what} found`;
    case "NotReadableError":
      return `The ${what} is in use by another app`;
    default:
      return `Could not open the ${what}: ${e?.message || e}`;
  }
}

/**
 * Shows the camera in `#camera-dialog` until a photo is taken or the
 * dialog is closed: the photo as a Blob, or null if cancelled. Rejects if
 * the camera cannot be opened or the photo cannot be encoded.
 */
export async function capturePhoto() {
  const dialog = $("#camera-dialog");
  const video = $("#camera-video");
  const snap = $("#camera-snap");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" } }, // the back camera, if any
    audio: false,
  });
  video.srcObject = stream;
  return new Promise((resolve, reject) => {
    let photo = null;
    let failure = null;
    const onSnap = () => {
      if (!video.videoWidth) return; // no frame yet
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      canvas.toBlob(
        (blob) => {
          photo = blob;
          if (!blob) failure = new Error("This browser could not encode the photo");
          dialog.close();
        },
        "image/jpeg",
        0.92,
      );
    };
    dialog.addEventListener(
      "close",
      () => {
        snap.removeEventListener("click", onSnap);
        for (const track of stream.getTracks()) track.stop();
        video.srcObject = null;
        if (failure) reject(failure);
        else resolve(photo);
      },
      { once: true },
    );
    snap.addEventListener("click", onSnap);
    dialog.showModal();
  });
}
