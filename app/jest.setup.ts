// Jest-wide mocks for native modules without a JS fallback.
jest.mock("@react-native-async-storage/async-storage", () =>
  require("@react-native-async-storage/async-storage/jest/async-storage-mock"),
);

jest.mock("expo-secure-store", () => {
  const store = new Map<string, string>();
  return {
    getItemAsync: jest.fn(async (k: string) => store.get(k) ?? null),
    setItemAsync: jest.fn(async (k: string, v: string) => void store.set(k, v)),
    deleteItemAsync: jest.fn(async (k: string) => void store.delete(k)),
    __reset: () => store.clear(),
  };
});

// Tests never touch the network: api/ tests pass their own fetch.
jest.mock("expo/fetch", () => ({
  fetch: jest.fn(async () => {
    throw new Error("network access in a test");
  }),
}));

jest.mock("react-native-keyboard-controller", () => require("react-native-keyboard-controller/jest"));

// Players and the photo library have no JS fallback; the chat's media views
// and Save are exercised on devices.
jest.mock("expo-media-library", () => ({
  Asset: { create: jest.fn(async () => ({ id: "asset" })) },
  requestPermissionsAsync: jest.fn(async () => ({ granted: true })),
}));

jest.mock("expo-audio", () => ({
  useAudioPlayer: () => ({ play: jest.fn(), pause: jest.fn(), seekTo: jest.fn(async () => {}) }),
  useAudioPlayerStatus: () => ({ playing: false, currentTime: 0, duration: 0 }),
}));

jest.mock("expo-video", () => ({
  useVideoPlayer: () => ({}),
  VideoView: () => null,
}));
