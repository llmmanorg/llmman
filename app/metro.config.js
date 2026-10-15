// Learn more https://docs.expo.dev/guides/customizing-metro
const { getDefaultConfig } = require("expo/metro-config");
const path = require("node:path");

const config = getDefaultConfig(__dirname);

// Native-only libraries the web page has no use for, swapped for what they
// amount to in a browser (web/shims/). Each pulled in Reanimated, about
// half of the page's JavaScript, to do nothing there: the browser moves
// its own content for the keyboard, and the web has no native gestures.
const WEB_SHIMS = {
  "react-native-gesture-handler": "web/shims/gestureHandler.tsx",
  "react-native-keyboard-controller": "web/shims/keyboardController.tsx",
};

const resolveRequest = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  const shim = platform === "web" ? WEB_SHIMS[moduleName] : undefined;
  if (shim) return { type: "sourceFile", filePath: path.join(__dirname, shim) };
  return (resolveRequest ?? context.resolveRequest)(context, moduleName, platform);
};

module.exports = config;
