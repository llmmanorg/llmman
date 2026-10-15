// The web's own dialogs: alerts, confirmations and the key prompt, drawn by
// <DialogHost> in the page. window.alert/confirm/prompt are no-ops in some
// web views (llmman-desktop's WKWebView has no UI delegate: prompt returns
// null, confirm false), so the web app never relies on them. Native keeps
// the system's Alert (lib/alert.ts).

export interface DialogButton {
  text: string;
  style?: "default" | "cancel" | "destructive";
}

export interface DialogSpec {
  title: string;
  message?: string;
  /** A text field, e.g. for a key; its text comes back with the choice. */
  input?: { label?: string; placeholder?: string; secure?: boolean };
  /** A line under the field: where to find what it asks for. */
  hint?: string;
  /** Shown with the llmman mark: a question from llmman itself, like its key. */
  brand?: boolean;
  buttons: DialogButton[];
}

/** Which button was chosen (its index; -1 when dismissed), and the field's text. */
export interface DialogResult {
  button: number;
  text: string;
}

export interface OpenDialog extends DialogSpec {
  id: number;
}

let next = 1;
let queue: { dialog: OpenDialog; resolve: (r: DialogResult) => void }[] = [];
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

/** Shows a dialog after any already open; resolves when one of its buttons is chosen. */
export function showDialog(spec: DialogSpec): Promise<DialogResult> {
  return new Promise((resolve) => {
    queue = [...queue, { dialog: { ...spec, id: next++ }, resolve }];
    emit();
  });
}

/** The dialog on screen, if any. */
export function currentDialog(): OpenDialog | undefined {
  return queue[0]?.dialog;
}

/** Closes dialog `id` with `result`, showing the next one. */
export function resolveDialog(id: number, result: DialogResult): void {
  const entry = queue.find((q) => q.dialog.id === id);
  if (!entry) return;
  queue = queue.filter((q) => q !== entry);
  emit();
  entry.resolve(result);
}

export function subscribeDialogs(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** The button Escape or a tap outside means: the cancel one, else none (-1). */
export function cancelIndex(spec: DialogSpec): number {
  return spec.buttons.findIndex((b) => b.style === "cancel");
}

/**
 * The button Enter means (the primary action): the last one that neither
 * cancels nor destroys, else the last destructive one when that is the only
 * action, so Enter never picks "…and its chats" over a safer choice.
 */
export function primaryIndex(spec: DialogSpec): number {
  const last = (ok: (b: DialogButton) => boolean) => {
    for (let i = spec.buttons.length - 1; i >= 0; i--) if (ok(spec.buttons[i]!)) return i;
    return -1;
  };
  const safe = last((b) => b.style !== "cancel" && b.style !== "destructive");
  return safe >= 0 ? safe : last((b) => b.style === "destructive");
}
