// https://docs.expo.dev/guides/using-eslint/
const { defineConfig } = require("eslint/config");
const expoConfig = require("eslint-config-expo/flat");
const prettier = require("eslint-plugin-prettier/recommended");

module.exports = defineConfig([
  expoConfig,
  prettier,
  {
    ignores: ["dist/*", "ios/*", "android/*", ".expo/*", "coverage/*", "components/terminal/*.generated.ts"],
  },
  {
    // jest.mock factories must use require().
    files: ["jest.setup.ts", "**/__tests__/**"],
    rules: { "@typescript-eslint/no-require-imports": "off" },
  },
  {
    // Screens never talk HTTP: everything goes through api/.
    files: ["app/**", "components/**", "lib/**"],
    rules: {
      "no-restricted-globals": ["error", { name: "fetch", message: "Use the api/ module." }],
      "no-restricted-imports": [
        "error",
        { paths: [{ name: "expo/fetch", message: "Use the api/ module." }] },
      ],
    },
  },
]);
