package org.llmman.app

import android.content.Context
import android.system.ErrnoException
import android.system.Os
import android.util.Base64
import android.util.Log
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.SecureRandom

/**
 * One `llmman serve` child process and the filesystem it needs.
 *
 * Every executable comes from the APK's native-lib directory: since API 29
 * an app may not execve() anything under its writable storage, so build.gradle
 * ships `llmman`, its Go shim and llama.cpp as `lib*.so` there. The daemon
 * finds `llama-server` through a symlink on PATH (the kernel resolves the
 * link; SELinux checks the target), and the shared objects through
 * `LD_LIBRARY_PATH` (llama.cpp's Android build carries no RUNPATH).
 *
 * Loopback is shared by every app on the phone, and the daemon offers a
 * shell, so it runs with an API key ([apiKey]) that only this app knows:
 * the WebView is handed it at document start, `llmman` in the Shell tab
 * gets it as `LLMMAN_API_KEY`, and anything else on 127.0.0.1 gets a 401.
 */
class Daemon(private val context: Context) {
    private val nativeDir = File(context.applicationInfo.nativeLibraryDir)

    /** `$HOME`: the daemon puts its store under `~/.local/share/llmman`. */
    private val home = File(context.filesDir, "home")
    private val binDir = File(context.filesDir, "bin")
    private val tmpDir = File(context.cacheDir, "tmp")
    val logFile = logFile(context)

    @Volatile
    var process: Process? = null
        private set

    /** Spawns the daemon. Throws if the exec fails outright. */
    @Throws(IOException::class)
    fun start(): Process {
        prepareFilesystem()
        val exe = File(nativeDir, "libllmman.so")
        if (!exe.canExecute()) {
            throw IOException("${exe.path} is not executable; is extractNativeLibs on?")
        }
        val pb = ProcessBuilder(exe.path, "serve")
            .directory(home)
            .redirectErrorStream(true)
        pb.environment().apply {
            put("HOME", home.path)
            put("TMPDIR", tmpDir.path)
            // Only ever our own llama-server: no download, no container probe.
            put("LLMMAN_RUNTIME", "path")
            put("PATH", listOf(binDir.path, "/system/bin", "/system/xbin").joinToString(":"))
            put("LD_LIBRARY_PATH", nativeDir.path)
            // The web UI's Shell tab; there is no /bin/sh on Android.
            put("SHELL", "/system/bin/sh")
            put("LLMMAN_SHELL", "/system/bin/sh")
            put("LLMMAN_HOST", "127.0.0.1:$PORT")
            val key = apiKey(context)
            put("LLMMAN_API_KEYS", key)
            // The CLI's own key, so `llmman ps` works in the Shell tab.
            put("LLMMAN_API_KEY", key)
            // Bionic's getpwuid() gives app UIDs a placeholder; tools that
            // ask for a user name (Go's os/user, git) prefer these.
            put("USER", "llmman")
            put("LOGNAME", "llmman")
        }
        rotateLog()
        Log.i(TAG, "exec ${pb.command()} in ${home.path}")
        val proc = pb.start()
        process = proc
        Thread({ pump(proc) }, "llmman-log").apply { isDaemon = true }.start()
        return proc
    }

    /** SIGTERM: `serve` unloads models and exits; SIGKILL after a grace period. */
    fun stop() {
        val proc = process ?: return
        proc.destroy()
        val deadline = System.currentTimeMillis() + STOP_GRACE_MS
        while (proc.isAlive && System.currentTimeMillis() < deadline) {
            Thread.sleep(50)
        }
        if (proc.isAlive) {
            Log.w(TAG, "daemon ignored SIGTERM, killing")
            proc.destroyForcibly()
        }
        process = null
    }

    /** `GET /api/version` answers: the router is up, not merely the socket. */
    fun isHealthy(): Boolean {
        val conn = try {
            (URL("${BASE_URL}api/version").openConnection() as HttpURLConnection).apply {
                connectTimeout = 1000
                readTimeout = 1000
                requestMethod = "GET"
                setRequestProperty("Authorization", "Bearer ${apiKey(context)}")
            }
        } catch (e: IOException) {
            return false
        }
        return try {
            conn.responseCode == 200
        } catch (e: IOException) {
            false
        } finally {
            conn.disconnect()
        }
    }

    private fun prepareFilesystem() {
        for (dir in listOf(home, binDir, tmpDir)) {
            if (!dir.isDirectory && !dir.mkdirs()) throw IOException("mkdir ${dir.path}")
        }
        symlink(File(nativeDir, "libllama-server.so"), File(binDir, "llama-server"))
        // For the web UI's Shell tab: `llmman ps`, `llmman list`, …
        symlink(File(nativeDir, "libllmman.so"), File(binDir, "llmman"))
    }

    /** Idempotent: re-pointed on every start, so an app update is picked up. */
    private fun symlink(target: File, link: File) {
        if (!target.isFile) throw IOException("missing ${target.path}; the APK lacks llama.cpp")
        try {
            val current = try {
                Os.readlink(link.path)
            } catch (e: ErrnoException) {
                null
            }
            if (current == target.path) return
            link.delete()
            Os.symlink(target.path, link.path)
        } catch (e: ErrnoException) {
            throw IOException("symlink ${link.path} -> ${target.path}", e)
        }
    }

    private fun rotateLog() {
        if (logFile.length() > LOG_ROTATE_BYTES) {
            logFile.renameTo(File(logFile.path + ".1"))
        }
    }

    private fun pump(proc: Process) {
        try {
            logFile.appendText("---- llmman serve started ${java.util.Date()} ----\n")
            proc.inputStream.bufferedReader().useLines { lines ->
                for (line in lines) {
                    Log.i(TAG, line)
                    logFile.appendText(line + "\n")
                }
            }
        } catch (e: IOException) {
            Log.w(TAG, "log pump ended: $e")
        }
    }

    companion object {
        const val TAG = "llmman"
        const val PORT = 17434
        const val BASE_URL = "http://127.0.0.1:$PORT/"
        private const val STOP_GRACE_MS = 5000L
        private const val LOG_ROTATE_BYTES = 2L * 1024 * 1024
        fun logFile(context: Context): File = File(context.filesDir, "serve.log")

        private val keyLock = Any()

        /**
         * This install's API key, generated once (256 bits, base64url) into
         * the app's private files — which no other app can read — and
         * reused for the life of the install. The Service and the Activity
         * both call this; the lock keeps the first call the only writer.
         */
        fun apiKey(context: Context): String = synchronized(keyLock) {
            val file = File(context.filesDir, "api-key")
            val existing = if (file.isFile) file.readText().trim() else ""
            if (existing.isNotEmpty()) return existing
            val bytes = ByteArray(32).also { SecureRandom().nextBytes(it) }
            val key = Base64.encodeToString(bytes, Base64.URL_SAFE or Base64.NO_PADDING or Base64.NO_WRAP)
            val tmp = File(file.path + ".tmp")
            tmp.writeText(key)
            if (!tmp.renameTo(file)) throw IOException("rename ${tmp.path}")
            key
        }
    }
}
