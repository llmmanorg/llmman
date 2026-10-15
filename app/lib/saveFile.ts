// Hands a file to the user: on a phone, the share sheet (Files, AirDrop,
// Drive…); on the web, a download (saveFile.web.ts).

import { File, Paths } from "expo-file-system";
import * as Sharing from "expo-sharing";

/** `content` is text, or base64 when `base64` is set. */
export async function saveFile(
  name: string,
  content: string,
  mimeType: string,
  { base64 = false }: { base64?: boolean } = {},
): Promise<void> {
  const file = new File(Paths.cache, name);
  if (file.exists) file.delete();
  file.create();
  file.write(content, base64 ? { encoding: "base64" } : {});
  await Sharing.shareAsync(file.uri, { mimeType, dialogTitle: name });
}
