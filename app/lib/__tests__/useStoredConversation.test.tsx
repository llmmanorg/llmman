// A screen opened by its address (after a reload) reads its chat from storage.

import AsyncStorage from "@react-native-async-storage/async-storage";
import { renderHook, waitFor } from "@testing-library/react-native";

import { useStoredConversation } from "../chatStore";
import { newConversation, saveConversation } from "../conversations";

jest.mock("expo-crypto", () => {
  let n = 0;
  return { randomUUID: () => `stored-${++n}` };
});

beforeEach(() => AsyncStorage.clear());

it("reads a saved chat that is not in the store yet", async () => {
  const conv = newConversation("http://a:1", "m");
  conv.messages.push({ role: "user", content: "hi", at: 1 });
  await saveConversation(conv);

  const { result } = renderHook(() => useStoredConversation(conv.id));
  await waitFor(() => expect(result.current.conv?.id).toBe(conv.id));
  expect(result.current.missing).toBe(false);
});

it("says so when there is no such chat", async () => {
  const { result } = renderHook(() => useStoredConversation("gone"));
  await waitFor(() => expect(result.current.missing).toBe(true));
  expect(result.current.conv).toBeNull();
});
