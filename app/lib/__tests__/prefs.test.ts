import { greeting, parsePrefs } from "@/lib/prefs";

test("reads the web UI's settings, keeping its other keys", () => {
  const raw = JSON.stringify({
    theme: "dark",
    name: "Ricky",
    sendWith: "mod-enter",
    model: "m",
    sidebarCollapsed: true,
  });
  const { prefs, stored } = parsePrefs(raw);
  // Its collapsed sidebar is the app's too.
  expect(prefs).toEqual({ theme: "dark", name: "Ricky", sendWith: "mod-enter", sidebarCollapsed: true });
  expect(stored.model).toBe("m");
});

test("falls back to the defaults for bad values", () => {
  expect(parsePrefs("not json").prefs).toEqual({
    theme: "system",
    name: "",
    sendWith: "enter",
    sidebarCollapsed: false,
  });
  expect(parsePrefs('{"theme":"blue","name":3,"sendWith":"x","sidebarCollapsed":"yes"}').prefs).toEqual({
    theme: "system",
    name: "",
    sendWith: "enter",
    sidebarCollapsed: false,
  });
});

test("greets by the time of day, with the name when there is one", () => {
  expect(greeting("", 9)).toBe("Good morning");
  expect(greeting("Ricky", 14)).toBe("Good afternoon, Ricky");
  expect(greeting("Ricky", 22)).toBe("Good evening, Ricky");
  expect(greeting("", 3)).toBe("Good evening");
});
