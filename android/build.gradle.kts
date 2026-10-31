// Android app: the web UI in a WebView, talking to an embedded `llmman serve`.
// See docs/android.md. Everything platform-specific lives in app/.
plugins {
    id("com.android.application") version "9.5.0" apply false
    id("org.jetbrains.kotlin.android") version "2.4.21" apply false
}
