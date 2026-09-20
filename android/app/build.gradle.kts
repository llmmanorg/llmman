import java.io.File
import java.util.Base64

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// ── Versioning: one source of truth, Cargo.toml (CI writes the release
// version into it via packaging/version.sh --apply before building). ──────
val repoRoot: File = rootProject.projectDir.parentFile
val cargoVersion: String = repoRoot.resolve("Cargo.toml").readLines()
    .first { it.trim().startsWith("version = ") }
    .substringAfter('"').substringBefore('"')

/** `MAJOR.MINOR.PATCH` -> a strictly increasing int. PATCH is the commit
 *  count (see packaging/version.sh), so give it six digits of room. */
fun versionCodeOf(version: String): Int {
    val (major, minor, patch) = version.split('.').map { it.toInt() }
    return major * 100_000_000 + minor * 1_000_000 + patch
}

// llama.cpp is pinned by the same file the daemon's `--runtime bin` uses.
val llamaCppRelease: String = repoRoot.resolve("LLAMA_CPP_RELEASE").readText().trim()
val llamaCppTarballName = "llama-$llamaCppRelease-bin-android-arm64.tar.gz"
val llamaCppTarball: Provider<RegularFile> =
    layout.buildDirectory.file("llama.cpp/$llamaCppTarballName")

// Everything an Android app may execute must arrive through the APK's
// native-lib directory (W^X: since API 29 execve() from app-writable
// storage is denied). So the daemon, its Go shim and llama.cpp are all
// staged here as lib*.so and installed with extractNativeLibs=true.
val nativeLibsDir: Provider<Directory> = layout.buildDirectory.dir("jniLibs/arm64-v8a")
val generatedAssetsDir: Provider<Directory> = layout.buildDirectory.dir("generated-assets")

val cargoTarget = "aarch64-linux-android"
val cargoOutDir: File = repoRoot.resolve("target/$cargoTarget/release")

val releaseKeystoreBase64: Provider<String> = providers.environmentVariable("LLMMAN_ANDROID_KEYSTORE")
val releaseKeystore: Provider<RegularFile> = layout.buildDirectory.file("release.keystore")

android {
    namespace = "org.llmman.app"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "org.llmman.app"
        // 28 = the llama.cpp Android release's own minimum (NDK
        // __ANDROID_API__ note); nothing in the app needs newer.
        minSdk = 28
        targetSdk = 35
        versionCode = versionCodeOf(cargoVersion)
        versionName = cargoVersion
        ndk { abiFilters += "arm64-v8a" }
    }

    // CI passes a release key through LLMMAN_ANDROID_* (see ci.yml's
    // android job). Without one the release APK is debug-signed: it still
    // installs by sideload, just not on top of a release-signed install.
    // The keystore file itself is written by the writeKeystore task below,
    // not here: configuration is skipped on a configuration-cache hit, so
    // a file written during it would be missing after `clean`.
    if (releaseKeystoreBase64.orNull?.isNotBlank() == true) {
        signingConfigs.create("release") {
            storeFile = releaseKeystore.get().asFile
            storePassword = providers.environmentVariable("LLMMAN_ANDROID_KEYSTORE_PASSWORD").orNull
            keyAlias = providers.environmentVariable("LLMMAN_ANDROID_KEY_ALIAS").orNull
            keyPassword = providers.environmentVariable("LLMMAN_ANDROID_KEY_PASSWORD").orNull
        }
    }

    buildTypes {
        release {
            // Nothing to shrink: one Activity and one Service of Kotlin.
            // Leaving R8 off keeps the JavascriptInterface and Service
            // names stable without a keep-rules file to maintain.
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    packaging {
        jniLibs {
            // Extract lib*.so to disk at install time so they can be exec'd;
            // the default (mmap straight from the APK) has no file path.
            useLegacyPackaging = true
        }
    }

    sourceSets["main"].jniLibs.srcDir(layout.buildDirectory.dir("jniLibs"))
    sourceSets["main"].assets.srcDir(generatedAssetsDir)

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.webkit:webkit:1.12.1")
}

// ── llama.cpp: fetch the pinned Android release and stage what
// llama-server needs. curl (not the JVM) so proxies/CA overrides in the
// environment apply, the same way build.rs fetches the web UI's assets. ──
val fetchLlamaCpp by tasks.registering(Exec::class) {
    description = "Download llama.cpp $llamaCppRelease for android-arm64"
    val out = llamaCppTarball.get().asFile
    outputs.file(out)
    onlyIf { !out.exists() }
    doFirst { out.parentFile.mkdirs() }
    commandLine(
        "curl", "-fsSL", "--retry", "3", "-o", out.path,
        "https://github.com/ggml-org/llama.cpp/releases/download/$llamaCppRelease/$llamaCppTarballName",
    )
}

val stageLlamaCpp by tasks.registering(Copy::class) {
    description = "Stage llama-server and its libraries into jniLibs"
    dependsOn(fetchLlamaCpp)
    val tar = tarTree(resources.gzip(llamaCppTarball))
    // The launcher plus its DT_NEEDED closure and the ggml-cpu variants
    // libggml.so dlopens at runtime; the CLI/bench/rpc tools stay out.
    from(tar) {
        include(
            "*/libggml-base.so", "*/libggml.so", "*/libggml-cpu-*.so",
            "*/libllama.so", "*/libllama-common.so", "*/libmtmd.so",
            "*/libllama-server-impl.so",
        )
        eachFile { path = name }
    }
    // Only lib*.so files are packaged, so the launcher takes that name; the
    // Service exposes it on PATH as `llama-server` via a symlink.
    from(tar) {
        include("*/llama-server")
        rename { "libllama-server.so" }
        eachFile { path = name }
    }
    into(nativeLibsDir)
    includeEmptyDirs = false
}

val stageLlamaCppLicense by tasks.registering(Copy::class) {
    dependsOn(fetchLlamaCpp)
    from(tarTree(resources.gzip(llamaCppTarball))) {
        include("*/LICENSE")
        rename { "llama.cpp.LICENSE" }
        eachFile { path = name }
    }
    into(generatedAssetsDir.map { it.dir("licenses") })
    includeEmptyDirs = false
}

// ── llmman itself: cross-compiled with cargo-ndk. Always runs; cargo's own
// up-to-date check is the authority on whether anything rebuilds. ─────────
val buildLlmman by tasks.registering(Exec::class) {
    description = "cargo ndk build --release for $cargoTarget"
    workingDir = repoRoot
    outputs.upToDateWhen { false }
    outputs.file(cargoOutDir.resolve("llmman"))
    environment("ANDROID_HOME", android.sdkDirectory.path)
    environment("ANDROID_NDK_HOME", android.ndkDirectory.path)
    // rustup's cargo (which owns the Android std) ahead of any distro one.
    val rustupBin = File(System.getProperty("user.home"), ".cargo/bin")
    if (rustupBin.isDirectory) {
        environment("PATH", rustupBin.path + File.pathSeparator + System.getenv("PATH"))
    }
    commandLine("cargo", "ndk", "-t", "arm64-v8a", "--platform", "28", "build", "--release")
}

val stageLlmman by tasks.registering {
    description = "Stage the llmman binary and Go shim into jniLibs"
    dependsOn(buildLlmman)
    val dest = nativeLibsDir
    val out = cargoOutDir
    // With outputs alone Gradle would call this up-to-date while cargo had
    // rebuilt the binary underneath it; the inputs are what cargo wrote.
    inputs.file(out.resolve("llmman"))
    inputs.files(fileTree(out.resolve("build")) { include("llmman-*/out/libllmman_shim.so") })
    outputs.dir(dest)
    doLast {
        val destDir = dest.get().asFile.also { it.mkdirs() }
        out.resolve("llmman").copyTo(destDir.resolve("libllmman.so"), overwrite = true)
        // build.rs writes the c-shared shim to its OUT_DIR, whose hash
        // suffix is not knowable here; take the newest if several exist.
        val shim = out.resolve("build").listFiles { f -> f.name.startsWith("llmman-") }
            ?.map { it.resolve("out/libllmman_shim.so") }
            ?.filter { it.isFile }
            ?.maxByOrNull { it.lastModified() }
            ?: error("libllmman_shim.so not found under $out/build — did the Go shim build?")
        shim.copyTo(destDir.resolve("libllmman_shim.so"), overwrite = true)
    }
}

// ── Release keystore from the environment, as a task so it exists
// whenever signing runs (see the signingConfigs comment). ─────────────────
val writeKeystore by tasks.registering {
    description = "Decode LLMMAN_ANDROID_KEYSTORE into build/"
    val encoded = releaseKeystoreBase64
    val target = releaseKeystore
    onlyIf { encoded.orNull?.isNotBlank() == true }
    inputs.property("keystore", encoded).optional(true)
    outputs.file(target)
    doLast {
        val out = target.get().asFile
        out.parentFile.mkdirs()
        out.writeBytes(Base64.getDecoder().decode(encoded.get().trim()))
    }
}

tasks.named("preBuild") {
    dependsOn(stageLlamaCpp, stageLlamaCppLicense, stageLlmman, writeKeystore)
}
