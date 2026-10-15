import type { ExpoConfig } from "expo/config";

// The manatee's backdrop, shared with scripts/make-icons.mjs.
import brand from "./assets/brand.json";

const config: ExpoConfig = {
  name: "llmman",
  slug: "llmman-mobile",
  version: "1.0.0",
  // All orientations: iPad apps are expected to rotate, and landscape helps on a phone too.
  orientation: "default",
  icon: "./assets/icon.png",
  userInterfaceStyle: "automatic",
  scheme: "llmman",
  ios: {
    // Light on the pale teal, dark as the mark alone on the system's dark
    // backdrop, tinted as a grayscale mark the system colours.
    icon: {
      light: "./assets/icon.png",
      dark: "./assets/icon-dark.png",
      tinted: "./assets/icon-tinted.png",
    },
    supportsTablet: true,
    bundleIdentifier: "org.llmman.mobile",
    infoPlist: {
      // llmman serve speaks plain HTTP on the LAN: allow it for local
      // addresses only, not the internet.
      NSAppTransportSecurity: { NSAllowsLocalNetworking: true },
      NSLocalNetworkUsageDescription:
        "llmman connects to the computer running llmman serve on your local network.",
    },
  },
  android: {
    package: "org.llmman.mobile",
    adaptiveIcon: {
      backgroundColor: brand.background,
      foregroundImage: "./assets/android-icon-foreground.png",
      monochromeImage: "./assets/android-icon-monochrome.png",
    },
    predictiveBackGestureEnabled: false,
  },
  web: {
    // One page: llmman serve answers only `/` with index.html.
    output: "single",
    favicon: "./assets/favicon.png",
    name: "llmman",
    themeColor: brand.background,
  },
  plugins: [
    // On the web each route is its own chunk, loaded when it is opened: the
    // page starts with the chat alone, not the shell, models or settings.
    // Native apps load routes up front, as before.
    ["expo-router", { asyncRoutes: { web: true, default: false } }],
    "expo-secure-store",
    [
      "expo-splash-screen",
      {
        image: "./assets/llmman-mark.png",
        imageWidth: 160,
        resizeMode: "contain",
        backgroundColor: brand.background,
        dark: { image: "./assets/llmman-mark.png", backgroundColor: brand.darkBackground },
      },
    ],
    [
      "expo-image-picker",
      {
        photosPermission: "llmman attaches the photos you choose to a chat.",
        cameraPermission: "llmman attaches the photos you take to a chat.",
        microphonePermission: false,
      },
    ],
    [
      "expo-media-library",
      {
        photosPermission: "llmman saves the images and videos it generates to your photo library.",
        savePhotosPermission: "llmman saves the images and videos it generates to your photo library.",
        isAccessMediaLocationEnabled: false,
        granularPermissions: ["photo", "video"],
      },
    ],
    [
      "expo-build-properties",
      {
        // llmman serve speaks plain HTTP on the LAN. Android's network
        // security config matches domains only, not IP ranges, so cleartext
        // cannot be limited to private addresses and is allowed app-wide
        // (iOS limits it to local networks, above).
        android: { usesCleartextTraffic: true },
        // iOS 27 requires the UIScene lifecycle: without it UIKit stops the
        // app at launch. Expo's scene delegate hosts the React Native root
        // view and forwards opened URLs to Linking.
        // SDK 58's template has this built in; then the option goes.
        ios: { enableSceneSupport: true },
      },
    ],
    // Generated video and sound play in the chat, in the foreground only;
    // nothing records.
    ["expo-video", { supportsBackgroundPlayback: false, supportsPictureInPicture: false }],
    [
      "expo-audio",
      { microphonePermission: false, recordAudioAndroid: false, enableBackgroundPlayback: false },
    ],
    // Font or display size changes must not recreate the activity.
    "./plugins/withActivityConfigChanges",
  ],
  // Set by scripts/export-web.mjs only: a placeholder base path it swaps
  // for one found at run time, so the page works under any path prefix.
  experiments: process.env.LLMMAN_EXPORT_BASE ? { baseUrl: process.env.LLMMAN_EXPORT_BASE } : undefined,
  // No typed routes: they are generated into .expo/ by a running dev
  // server, so `tsc` on a fresh checkout would see a different program.
};

export default config;
