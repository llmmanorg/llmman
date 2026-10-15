// Random ids for servers and chats. expo-crypto's randomUUID is the
// browser's crypto.randomUUID on the web, which exists only in a secure
// context (https or localhost): a page served by llmman at http://<LAN IP>
// has none, so there the id comes from crypto.getRandomValues instead.

import { randomUUID } from "expo-crypto";

export function newId(): string {
  try {
    return randomUUID();
  } catch {
    return uuidFrom(globalThis.crypto.getRandomValues(new Uint8Array(16)));
  }
}

/** A version 4 UUID from 16 random bytes. */
export function uuidFrom(bytes: Uint8Array): string {
  const b = Uint8Array.from(bytes);
  b[6] = (b[6]! & 0x0f) | 0x40;
  b[8] = (b[8]! & 0x3f) | 0x80;
  const hex = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
