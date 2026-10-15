# llmman app

The web UI `llmman serve` serves at `/`, and the iOS and Android apps, built
from one Expo (React Native) source. What the page does is in
[docs/webui.md](../docs/webui.md).

```
npm ci                          # also downloads the mark and makes the icons
npm start                       # dev server: w opens a browser, i/a a simulator
npm run check                   # typecheck, lint, format, tests
npm run export-web -- ../webui  # the web UI llmman embeds; commit the result
```

CI fails when `webui/` is not the export of this source, so run the export
and commit `webui/` with any change here. To try a change in llmman without
rebuilding it, export to a directory and run
`LLMMAN_WEBUI_DIR=<that dir> llmman serve`.

The native apps build with `npx expo run:ios` or `npx expo run:android`
(Xcode or Android Studio), or with EAS; `ios/` and `android/` are generated,
never edited. Expo changes between SDKs: check the docs for the SDK in
`package.json` rather than memory.

`api/` is the only code that talks HTTP (lint enforces it), and API keys
live in the platform's secure store (localStorage on the web), never in
AsyncStorage or logs.
