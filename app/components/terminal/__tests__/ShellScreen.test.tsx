import { act, fireEvent, render, screen, waitFor } from "@testing-library/react-native";

import Shell from "@/app/(drawer)/shell";
import { openShellSocket } from "@/api/shell";
import type { ShellStatus } from "@/api/shell";
import type { TerminalProps } from "@/components/terminal/types";

const URL = "http://127.0.0.1:17434";

let mockShell: ShellStatus;
let mockToken: string | undefined;
let mockTerminal: TerminalProps | undefined;

jest.mock("expo-symbols", () => ({ SymbolView: () => null }));
jest.mock("expo-router", () => ({ useIsFocused: () => true }));
const mockServers = { active: { url: "http://127.0.0.1:17434" }, keyFor: async () => undefined };
jest.mock("@/lib/ServersContext", () => ({ useServers: () => mockServers }));
jest.mock("@/lib/DaemonContext", () => ({
  useDaemon: () => ({ features: { shell: mockShell }, reachable: true, checkReachable: () => {} }),
}));
jest.mock("@/lib/page", () => ({
  pageUrl: () => "http://127.0.0.1:17434",
  pageShellToken: () => mockToken,
  onPageShellToken: () => () => {},
}));
jest.mock("@/api/shell", () => ({ openShellSocket: jest.fn() }));
jest.mock("@/components/terminal/KeysBar", () => ({ KeysBar: () => null }));
jest.mock("@/components/terminal/Terminal", () => ({
  Terminal: (props: TerminalProps) => {
    const { useEffect, useRef } = jest.requireActual("react");
    const { Text: MockText } = jest.requireActual("react-native");
    // Once per mount, as the real one: reconnecting takes a remount.
    const connectAtMount = useRef(props.connect);
    mockTerminal = props;
    useEffect(() => void connectAtMount.current(), []);
    return <MockText>terminal</MockText>;
  },
}));

const open = openShellSocket as jest.MockedFunction<typeof openShellSocket>;

beforeEach(() => {
  open.mockReset();
  mockTerminal = undefined;
});

test("a daemon that needs a token, with none here: explains and opens nothing", async () => {
  mockShell = { enabled: true, endpoint: true, tokenRequired: true };
  mockToken = undefined;
  render(<Shell />);
  await act(async () => {});
  expect(screen.getByText("Open the shell link from llmman serve")).toBeTruthy();
  expect(screen.queryByText("terminal")).toBeNull();
  expect(open).not.toHaveBeenCalled();
});

test("a daemon older than the token: opens a shell without one", async () => {
  mockShell = { enabled: true, endpoint: true };
  mockToken = undefined;
  render(<Shell />);
  await screen.findByText("terminal");
  expect(open).toHaveBeenCalledWith(URL, undefined, undefined);
});

test("a token is offered to the daemon that needs it", async () => {
  mockShell = { enabled: true, endpoint: true, tokenRequired: true };
  mockToken = "tok";
  render(<Shell />);
  await screen.findByText("terminal");
  expect(open).toHaveBeenCalledWith(URL, undefined, "tok");
});

test("refused before attaching: a stale-token hint, and a new shell takes the newest token", async () => {
  mockShell = { enabled: true, endpoint: true, tokenRequired: true };
  mockToken = "old";
  render(<Shell />);
  await screen.findByText("terminal");
  expect(open).toHaveBeenLastCalledWith(URL, undefined, "old");
  act(() => mockTerminal!.onState("closed"));
  expect(screen.getByText(/each start makes a new token/)).toBeTruthy();

  mockToken = "new";
  fireEvent.press(screen.getByText("Open a new shell"));
  await waitFor(() => expect(open).toHaveBeenLastCalledWith(URL, undefined, "new"));
});
