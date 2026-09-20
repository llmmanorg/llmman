package org.llmman.app

import android.Manifest
import android.app.DownloadManager
import android.content.ActivityNotFoundException
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.view.View
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.URLUtil
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import java.io.File
import java.io.IOException

/**
 * The web UI (served by the embedded daemon) in a WebView, behind a status
 * screen until the daemon is healthy. Nothing here knows about models or
 * chat; that is all webui/ — this class is only what a browser would do
 * for it: navigation, downloads, file pickers, external links.
 */
class MainActivity : ComponentActivity() {
    private lateinit var webView: WebView
    private lateinit var status: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var progress: ProgressBar
    private lateinit var actions: View
    private lateinit var logScroll: ScrollView
    private lateinit var logText: TextView
    private lateinit var toggleLog: Button

    private var uiLoaded = false
    private var fileChooser: ValueCallback<Array<Uri>>? = null

    private val pickFiles = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        fileChooser?.onReceiveValue(uris.toTypedArray())
        fileChooser = null
    }

    private val requestNotifications =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    /** Bytes awaiting the ACTION_CREATE_DOCUMENT result (API 28 saves). */
    private var pendingSave: ByteArray? = null

    private val createDocument = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val bytes = pendingSave ?: return@registerForActivityResult
        pendingSave = null
        val uri = result.data?.data ?: return@registerForActivityResult
        Thread {
            try {
                contentResolver.openOutputStream(uri)?.use { it.write(bytes) } ?: throw IOException("cannot open $uri")
                runOnUiThread {
                    Toast.makeText(this, getString(R.string.download_saved, uri.lastPathSegment ?: ""), Toast.LENGTH_SHORT).show()
                }
            } catch (e: IOException) {
                Log.w(Daemon.TAG, "save to $uri failed", e)
                runOnUiThread { Toast.makeText(this, R.string.download_failed, Toast.LENGTH_SHORT).show() }
            }
        }.start()
    }

    private val stateListener: (LlmmanService.State) -> Unit = { onState(it) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        webView = findViewById(R.id.webview)
        status = findViewById(R.id.status)
        statusText = findViewById(R.id.status_text)
        progress = findViewById(R.id.progress)
        actions = findViewById(R.id.actions)
        logScroll = findViewById(R.id.log_scroll)
        logText = findViewById(R.id.log_text)
        toggleLog = findViewById(R.id.toggle_log)

        applyInsets()
        setupWebView()

        findViewById<Button>(R.id.retry).setOnClickListener {
            LlmmanService.start(this)
            if (LlmmanService.State.current is LlmmanService.State.Running) loadUi()
        }
        toggleLog.setOnClickListener {
            val show = logScroll.visibility != View.VISIBLE
            logScroll.visibility = if (show) View.VISIBLE else View.GONE
            toggleLog.setText(if (show) R.string.action_hide_log else R.string.action_show_log)
            if (show) refreshLog()
        }

        onBackPressedDispatcher.addCallback(this) {
            if (webView.visibility == View.VISIBLE && webView.canGoBack()) {
                webView.goBack()
            } else {
                // Keep the Activity (and its WebView state) around; the
                // service keeps the daemon up regardless.
                moveTaskToBack(true)
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            requestNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        LlmmanService.start(this)
        LlmmanService.State.observe(stateListener)
    }

    override fun onDestroy() {
        LlmmanService.State.unobserve(stateListener)
        webView.destroy()
        super.onDestroy()
    }

    /** Edge-to-edge is mandatory at targetSdk 35; pad for bars and the keyboard. */
    private fun applyInsets() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val root = findViewById<View>(R.id.root)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or
                    WindowInsetsCompat.Type.displayCutout() or
                    WindowInsetsCompat.Type.ime(),
            )
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
    }

    private fun setupWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            // IndexedDB (conversations) and localStorage (settings, API key).
            domStorageEnabled = true
            // Honour <meta name="viewport">.
            useWideViewPort = true
            loadWithOverviewMode = true
            // Generated speech plays without a second tap.
            mediaPlaybackRequiresUserGesture = false
            allowFileAccess = false
            allowContentAccess = false
        }
        webView.addJavascriptInterface(Bridge(), "LlmmanAndroid")
        val startScript = startScript(Daemon.apiKey(this))
        val startScriptInjected = if (WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) {
            WebViewCompat.addDocumentStartJavaScript(webView, startScript, setOf(Daemon.BASE_URL.trimEnd('/')))
            true
        } else {
            false
        }
        webView.webViewClient = object : WebViewClient() {
            override fun onPageStarted(view: WebView, url: String, favicon: Bitmap?) {
                // Older WebViews: before the page's own module scripts run
                // is the best available approximation of document start.
                if (!startScriptInjected && isOurOrigin(Uri.parse(url))) {
                    view.evaluateJavascript(startScript, null)
                }
            }

            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                if (isOurOrigin(request.url)) return false
                openExternally(request.url)
                return true
            }

            override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
                if (!request.isForMainFrame) return
                uiLoaded = false
                showStatus(getString(R.string.status_ui_failed), busy = false, showActions = true)
            }

            override fun onPageFinished(view: WebView, url: String) {
                if (uiLoaded && isOurOrigin(Uri.parse(url))) showWebView()
            }
        }
        webView.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                view: WebView,
                callback: ValueCallback<Array<Uri>>,
                params: FileChooserParams,
            ): Boolean {
                fileChooser?.onReceiveValue(null)
                fileChooser = callback
                val types = params.acceptTypes.filter { it.isNotBlank() }
                pickFiles.launch(if (types.size == 1) types[0] else "*/*")
                return true
            }

            override fun onConsoleMessage(message: ConsoleMessage): Boolean {
                Log.d(Daemon.TAG, "webui: ${message.message()} (${message.sourceId()}:${message.lineNumber()})")
                return true
            }
        }
        // The UI offers "Export chats" and per-image Download as blob: URLs,
        // which no system component can fetch: only the page can read them.
        // The WebView also drops the anchor's `download` name for those
        // (empty Content-Disposition), so DOWNLOAD_NAME_SCRIPT keeps it.
        webView.setDownloadListener { url, _, contentDisposition, mimeType, _ ->
            val name = URLUtil.guessFileName(url, contentDisposition, mimeType)
            if (url.startsWith("blob:")) {
                webView.evaluateJavascript(blobToBridgeScript(url, name, mimeType), null)
            } else {
                downloadViaManager(url, name, mimeType)
            }
        }
    }

    private fun onState(state: LlmmanService.State) {
        when (state) {
            LlmmanService.State.Starting -> showStatus(getString(R.string.status_starting))
            LlmmanService.State.Running -> if (!uiLoaded) loadUi()
            is LlmmanService.State.Restarting -> {
                uiLoaded = false
                // The service is already retrying; the log is there for a crash loop.
                showStatus(getString(R.string.status_failed) + " (exit ${state.exitCode})", showActions = true)
            }
            is LlmmanService.State.Failed -> {
                uiLoaded = false
                showStatus(state.reason, busy = false, showActions = true)
            }
            LlmmanService.State.Stopped -> {
                uiLoaded = false
                showStatus(getString(R.string.status_waiting), busy = false, showActions = true)
            }
        }
    }

    private fun loadUi() {
        uiLoaded = true
        showStatus(getString(R.string.status_waiting))
        if (webView.url?.let { isOurOrigin(Uri.parse(it)) } == true) {
            webView.reload()
        } else {
            webView.loadUrl(Daemon.BASE_URL)
        }
    }

    private fun showWebView() {
        status.visibility = View.GONE
        webView.visibility = View.VISIBLE
    }

    private fun showStatus(text: String, busy: Boolean = true, showActions: Boolean = false) {
        webView.visibility = View.INVISIBLE
        status.visibility = View.VISIBLE
        statusText.text = text
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        actions.visibility = if (showActions) View.VISIBLE else View.GONE
        if (showActions && logScroll.visibility == View.VISIBLE) refreshLog()
    }

    private fun refreshLog() {
        Thread {
            val text = LlmmanService.recentLog(this)
            runOnUiThread {
                logText.text = text
                logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
            }
        }.start()
    }

    private fun isOurOrigin(uri: Uri): Boolean =
        uri.scheme == "http" && uri.host == "127.0.0.1" && uri.port == Daemon.PORT

    private fun openExternally(uri: Uri) {
        try {
            startActivity(Intent(Intent.ACTION_VIEW, uri))
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, uri.toString(), Toast.LENGTH_SHORT).show()
        }
    }

    private fun downloadViaManager(url: String, name: String, mimeType: String?) {
        val request = DownloadManager.Request(Uri.parse(url))
            .setMimeType(mimeType)
            .setTitle(name)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, name)
        } else {
            // Below Q the public directory needs WRITE_EXTERNAL_STORAGE,
            // which the app does not ask for.
            request.setDestinationInExternalFilesDir(this, Environment.DIRECTORY_DOWNLOADS, name)
        }
        getSystemService(DownloadManager::class.java).enqueue(request)
    }

    /** Reads the blob in-page and hands it to [Bridge.saveBlob] as base64. */
    private fun blobToBridgeScript(url: String, fallbackName: String, mimeType: String?): String {
        val js = ::jsString
        return """
            (async () => {
              try {
                const name = (window.__llmmanDownloadName && window.__llmmanDownloadName(${js(url)})) || ${js(fallbackName)};
                const blob = await (await fetch(${js(url)})).blob();
                const buf = new Uint8Array(await blob.arrayBuffer());
                let bin = "";
                for (let i = 0; i < buf.length; i += 0x8000) {
                  bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
                }
                LlmmanAndroid.saveBlob(name, blob.type || ${js(mimeType ?: "")}, btoa(bin));
              } catch (e) {
                LlmmanAndroid.saveFailed(String(e));
              }
            })();
        """.trimIndent()
    }

    /** Exposed to the page as `LlmmanAndroid`. Only our own origin ever loads. */
    inner class Bridge {
        @JavascriptInterface
        fun saveBlob(name: String, mimeType: String, base64: String) {
            val bytes = try {
                Base64.decode(base64, Base64.DEFAULT)
            } catch (e: IllegalArgumentException) {
                saveFailed(e.toString())
                return
            }
            val safeName = File(name).name.ifBlank { "download" }
            val type = mimeType.ifBlank { "application/octet-stream" }
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
                // No MediaStore.Downloads before Q and no storage permission
                // requested: the user picks where it goes.
                pendingSave = bytes
                runOnUiThread {
                    createDocument.launch(
                        Intent(Intent.ACTION_CREATE_DOCUMENT)
                            .addCategory(Intent.CATEGORY_OPENABLE)
                            .setType(type)
                            .putExtra(Intent.EXTRA_TITLE, safeName),
                    )
                }
                return
            }
            Thread {
                try {
                    writeToDownloads(safeName, type, bytes)
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, getString(R.string.download_saved, safeName), Toast.LENGTH_SHORT).show()
                    }
                } catch (e: IOException) {
                    Log.w(Daemon.TAG, "save $safeName failed", e)
                    saveFailed(e.toString())
                }
            }.start()
        }

        @JavascriptInterface
        fun saveFailed(reason: String) {
            Log.w(Daemon.TAG, "download failed: $reason")
            runOnUiThread {
                Toast.makeText(this@MainActivity, R.string.download_failed, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private companion object {
        /**
         * Runs before any page script: stores the daemon's API key where
         * webui/api.js reads it (the same localStorage key it would write
         * after asking the user), then [DOWNLOAD_NAME_SCRIPT].
         */
        fun startScript(apiKey: String): String = """
            try {
              localStorage.setItem("llmman.apiKey:" + new URL(document.baseURI).pathname, ${jsString(apiKey)});
            } catch (e) {}
        """.trimIndent() + "\n" + DOWNLOAD_NAME_SCRIPT

        fun jsString(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""

        /**
         * Remembers the `download` attribute of every blob: anchor clicked,
         * by href, so the DownloadListener can name the file the way the
         * page meant to. webui creates detached anchors and calls click(),
         * so both the prototype method and real clicks are covered.
         */
        val DOWNLOAD_NAME_SCRIPT = """
            (() => {
              const names = new Map();
              const remember = (a) => {
                if (a && a.hasAttribute("download") && a.href.startsWith("blob:")) {
                  names.set(a.href, a.getAttribute("download"));
                }
              };
              const click = HTMLElement.prototype.click;
              HTMLAnchorElement.prototype.click = function () { remember(this); return click.call(this); };
              document.addEventListener("click", (e) => {
                remember(e.target instanceof Element ? e.target.closest("a[download]") : null);
              }, true);
              window.__llmmanDownloadName = (href) => names.get(href) || "";
            })();
        """.trimIndent()
    }

    /** Into the public Downloads collection, no permission needed on Q+. */
    @RequiresApi(Build.VERSION_CODES.Q)
    @Throws(IOException::class)
    private fun writeToDownloads(name: String, mimeType: String, bytes: ByteArray) {
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, name)
            put(MediaStore.Downloads.MIME_TYPE, mimeType)
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val resolver = contentResolver
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: throw IOException("MediaStore refused $name")
        resolver.openOutputStream(uri)?.use { it.write(bytes) } ?: throw IOException("cannot open $uri")
        values.clear()
        values.put(MediaStore.Downloads.IS_PENDING, 0)
        resolver.update(uri, values, null, null)
    }
}
