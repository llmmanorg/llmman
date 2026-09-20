// Android app: the web UI in a WebView, talking to an embedded `llmman serve`.
// See docs/android.md. Everything platform-specific lives in app/.
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.1.0" apply false
}
