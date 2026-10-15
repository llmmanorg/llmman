// The web: a download of the file, as the web UI's Export and Download do.

export async function saveFile(
  name: string,
  content: string,
  mimeType: string,
  { base64 = false }: { base64?: boolean } = {},
): Promise<void> {
  const blob = base64
    ? new Blob([Uint8Array.from(atob(content), (c) => c.charCodeAt(0))], { type: mimeType })
    : new Blob([content], { type: mimeType });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
