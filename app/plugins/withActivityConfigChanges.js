// Keeps the main activity alive through a font size or display size change
// (Android settings), as it already is through rotation and dark mode:
// recreating it would drop the user's screen, a streaming reply included.
// React Native re-reads the font scale and display metrics on the change.

const { AndroidConfig, withAndroidManifest } = require("expo/config-plugins");

const EXTRA = ["fontScale", "density"];

module.exports = function withActivityConfigChanges(config) {
  return withAndroidManifest(config, (cfg) => {
    const activity = AndroidConfig.Manifest.getMainActivityOrThrow(cfg.modResults);
    const current = (activity.$["android:configChanges"] ?? "").split("|").filter(Boolean);
    activity.$["android:configChanges"] = [...new Set([...current, ...EXTRA])].join("|");
    return cfg;
  });
};
