import { fireEvent, render, screen, waitFor } from "@testing-library/react-native";

import { ConnectForm } from "../ConnectForm";
import { checkServer } from "@/lib/connect";

jest.mock("expo-symbols", () => ({ SymbolView: () => null }));
jest.mock("@/lib/connect", () => ({ checkServer: jest.fn() }));

const mockSave = jest.fn(async () => {});
jest.mock("@/lib/ServersContext", () => ({
  useServers: () => ({ save: mockSave, servers: [{ id: "s", url: "http://10.0.0.77:17435", addedAt: 1 }] }),
}));

const check = checkServer as jest.MockedFunction<typeof checkServer>;

beforeEach(() => {
  mockSave.mockClear();
  check.mockReset();
});

const savedServer = { url: "http://10.0.0.77:17435", key: "bad-key" };

function fill(address: string, key?: string) {
  render(<ConnectForm onConnected={onConnected} />);
  fireEvent.changeText(screen.getByLabelText("Address"), address);
  if (key !== undefined) fireEvent.changeText(screen.getByLabelText("API key"), key);
}

const onConnected = jest.fn();
beforeEach(() => onConnected.mockClear());

it("never saves a known server's new key when the check fails", async () => {
  check.mockResolvedValue({ ok: false, field: "key", problem: "unauthorized", message: "rejected" });
  fill(savedServer.url, savedServer.key);
  fireEvent.press(screen.getByRole("button", { name: "Update server" }));
  expect(await screen.findByText("Wrong API key")).toBeTruthy();
  expect(mockSave).not.toHaveBeenCalled();
  expect(onConnected).not.toHaveBeenCalled();
});

it("says a key is needed when none was sent", async () => {
  check.mockResolvedValue({ ok: false, field: "key", problem: "unauthorized", message: "needs a key" });
  fill(savedServer.url);
  fireEvent.press(screen.getByRole("button", { name: "Update server" }));
  expect(await screen.findByText("API key needed")).toBeTruthy();
});

it("saves only what passed the check", async () => {
  check.mockResolvedValue({ ok: true, url: savedServer.url, key: "good", version: "0.1.0" });
  fill(savedServer.url, "good");
  fireEvent.press(screen.getByRole("button", { name: "Update server" }));
  await waitFor(() => expect(onConnected).toHaveBeenCalled());
  expect(mockSave).toHaveBeenCalledWith({ url: savedServer.url, key: "good" });
});
