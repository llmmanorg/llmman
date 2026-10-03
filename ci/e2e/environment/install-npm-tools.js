const fs = require("fs");

const manifest = JSON.parse(fs.readFileSync("ci/e2e/environment/manifest.json", "utf8"));

const shellQuote = (value) => `'${value.replace(/'/g, "'\\''")}'`;
const installSpecs = (installer) =>
  manifest.npm
    .filter((entry) => entry.installer === installer)
    .map((entry) => `${entry.name}@${entry.version}`);

const bun = installSpecs("bun");
const npm = installSpecs("npm");

if (bun.length) {
  console.log(`bun install -g ${bun.map(shellQuote).join(" ")}`);
}
if (npm.length) {
  console.log(`npm install -g ${npm.map(shellQuote).join(" ")}`);
}
