import { cancelIndex, currentDialog, primaryIndex, resolveDialog, showDialog } from "@/lib/dialog";

test("dialogs show one at a time, in order, and resolve with the choice and the text", async () => {
  const first = showDialog({
    title: "Key",
    input: { secure: true },
    buttons: [{ text: "Cancel", style: "cancel" }, { text: "Continue" }],
  });
  const second = showDialog({
    title: "Delete?",
    buttons: [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive" },
    ],
  });
  expect(currentDialog()?.title).toBe("Key");
  resolveDialog(currentDialog()!.id, { button: 1, text: "sk-x" });
  await expect(first).resolves.toEqual({ button: 1, text: "sk-x" });
  expect(currentDialog()?.title).toBe("Delete?");
  resolveDialog(currentDialog()!.id, { button: -1, text: "" });
  await expect(second).resolves.toEqual({ button: -1, text: "" });
  expect(currentDialog()).toBeUndefined();
});

test("Enter is the primary action, Escape the cancel one", () => {
  const spec = {
    title: "t",
    buttons: [
      { text: "Cancel", style: "cancel" as const },
      { text: "Remove", style: "destructive" as const },
    ],
  };
  expect(primaryIndex(spec)).toBe(1);
  expect(cancelIndex(spec)).toBe(0);
  expect(cancelIndex({ title: "t", buttons: [{ text: "OK" }] })).toBe(-1);
  // With a safe and a destructive action, Enter takes the safe one.
  const remove = {
    title: "Remove?",
    buttons: [
      { text: "Cancel", style: "cancel" as const },
      { text: "Remove server" },
      { text: "Remove server and its chats", style: "destructive" as const },
    ],
  };
  expect(primaryIndex(remove)).toBe(1);
});
