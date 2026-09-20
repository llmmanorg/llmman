package org.llmman.app

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.ServiceCompat
import java.io.IOException

/**
 * Foreground service that keeps `llmman serve` alive for as long as the
 * user wants a model server — including with the Activity gone, so a chat
 * agent on the LAN or a long pull is not cut off by the app being swiped
 * away. The notification's Stop action is the one way to end it.
 *
 * State is published on the main thread through [State.observe]; the
 * Activity uses it to decide when the WebView may load.
 */
class LlmmanService : Service() {
    sealed class State {
        data object Starting : State()
        data object Running : State()
        data class Restarting(val exitCode: Int, val delayMs: Long) : State()
        data class Failed(val reason: String) : State()
        data object Stopped : State()

        companion object {
            @Volatile
            var current: State = Stopped
                private set
            private val listeners = mutableSetOf<(State) -> Unit>()
            private val main = Handler(Looper.getMainLooper())

            fun observe(listener: (State) -> Unit) {
                main.post {
                    listeners += listener
                    listener(current)
                }
            }

            fun unobserve(listener: (State) -> Unit) {
                main.post { listeners -= listener }
            }

            internal fun publish(state: State) {
                main.post {
                    current = state
                    for (l in listeners.toList()) l(state)
                }
            }
        }
    }

    private lateinit var daemon: Daemon
    private var supervisor: Thread? = null

    @Volatile
    private var stopping = false

    override fun onCreate() {
        super.onCreate()
        daemon = Daemon(this)
        createChannel()
    }

    // FOREGROUND_SERVICE_TYPE_SPECIAL_USE is an API 34 constant;
    // ServiceCompat strips types the running OS does not know.
    @SuppressLint("InlinedApi")
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            shutdown()
            return START_NOT_STICKY
        }
        // Within 5 s of startForegroundService(), unconditionally.
        ServiceCompat.startForeground(
            this,
            NOTIFICATION_ID,
            notification(getString(R.string.notification_starting)),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
        )
        // A supervisor that gave up (Failed) or was stopped is dead but still
        // referenced; Retry must get a new one.
        if (supervisor?.isAlive != true) {
            stopping = false
            supervisor = Thread(::supervise, "llmman-supervisor").also { it.start() }
        }
        return START_STICKY
    }

    /** Start, wait for health, wait for exit, back off, repeat. */
    private fun supervise() {
        var backoffMs = 1000L
        while (!stopping) {
            State.publish(State.Starting)
            notify(getString(R.string.notification_starting))
            val proc = try {
                daemon.start()
            } catch (e: IOException) {
                Log.e(Daemon.TAG, "cannot start daemon", e)
                State.publish(State.Failed(e.message ?: e.toString()))
                notify(getString(R.string.notification_failed))
                // Failed stays the visible state; the loop ends without
                // publishing Stopped so the reason is what the user sees.
                return
            }
            val healthy = waitForHealth(proc)
            if (healthy) {
                backoffMs = 1000L
                State.publish(State.Running)
                notify(getString(R.string.notification_running))
            }
            val code = try {
                proc.waitFor()
            } catch (e: InterruptedException) {
                break
            }
            if (stopping) break
            Log.w(Daemon.TAG, "daemon exited with $code, restarting in ${backoffMs}ms")
            State.publish(State.Restarting(code, backoffMs))
            notify(getString(R.string.notification_restarting))
            try {
                Thread.sleep(backoffMs)
            } catch (e: InterruptedException) {
                break
            }
            backoffMs = (backoffMs * 2).coerceAtMost(MAX_BACKOFF_MS)
        }
        State.publish(State.Stopped)
    }

    private fun waitForHealth(proc: Process): Boolean {
        val deadline = System.currentTimeMillis() + STARTUP_BUDGET_MS
        while (proc.isAlive && !stopping && System.currentTimeMillis() < deadline) {
            if (daemon.isHealthy()) return true
            Thread.sleep(250)
        }
        return false
    }

    private fun shutdown() {
        stopping = true
        supervisor?.interrupt()
        Thread {
            daemon.stop()
            Handler(Looper.getMainLooper()).post {
                State.publish(State.Stopped)
                ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }.start()
    }

    override fun onDestroy() {
        stopping = true
        supervisor?.interrupt()
        // stop() waits up to STOP_GRACE_MS for the SIGTERM to take; not on
        // the main thread.
        Thread { daemon.stop() }.start()
        State.publish(State.Stopped)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel),
            NotificationManager.IMPORTANCE_LOW,
        ).apply { description = getString(R.string.notification_channel_description) }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun notification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val stop = PendingIntent.getService(
            this, 1,
            Intent(this, LlmmanService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(
                Notification.Action.Builder(null, getString(R.string.notification_stop), stop).build(),
            )
            .build()
    }

    private fun notify(text: String) {
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification(text))
    }

    companion object {
        const val ACTION_STOP = "org.llmman.app.STOP"
        private const val CHANNEL_ID = "server"
        private const val NOTIFICATION_ID = 1
        private const val MAX_BACKOFF_MS = 30_000L

        /** Matches the daemon's own client-side wait (daemon.rs STARTUP_BUDGET). */
        private const val STARTUP_BUDGET_MS = 60_000L

        fun start(context: Context) {
            context.startForegroundService(Intent(context, LlmmanService::class.java))
        }

        fun recentLog(context: Context): String {
            val file = Daemon.logFile(context)
            if (!file.isFile) return ""
            val lines = file.readLines()
            return lines.takeLast(200).joinToString("\n")
        }
    }
}
