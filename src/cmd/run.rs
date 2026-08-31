//! `llmman run` — interactive chat or one-shot prompt.
//!
//! Interactive mode uses a raw-mode readline ported directly from ollama's
//! readline package (readline/readline.go, readline/term.go). Paste
//! detection mirrors ollama exactly:
//!
//!   // ollama (Go)
//!   if i.Terminal.reader.Buffered() > 0 { draining = true }
//!
//!   // llmman (Rust)
//!   if !reader.buffer().is_empty() { draining = true; }
//!
//! While draining a paste, '\n' submits the line like Enter; otherwise
//! it's Ctrl-J multiline, same as ollama.
//!
//! Streamed responses are ported from ollama's `cmd/cmd.go`
//! (`generate`/`chat`, `displayResponse`, `thinkingOutputOpeningText`/
//! `...ClosingText`): word-wrapped at the terminal width, "Thinking..."
//! in dim grey/bold ANSI, sent over an async reqwest client racing
//! `tokio::signal::ctrl_c()` so Ctrl-C cancels only the in-flight turn
//! (mirrors ollama's `context.WithCancel` + `signal.Notify`) instead of
//! killing the process. In the interactive REPL, raw mode is released
//! for the duration of each response and re-entered before the next
//! prompt — mirrors `readline.Instance.Readline()`'s own enter/`defer`
//! restore — which is why Ctrl-C while typing is read as byte 3 (ISIG
//! off) but Ctrl-C during a response is a real SIGINT (ISIG back on).

use std::io::{self, IsTerminal, Write};
use std::time::Duration;

use anyhow::Context;
use clap::Args;
use futures::TryStreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;

use crate::daemon;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,
    /// Chat with MODEL at this hosted provider (openai, anthropic,
    /// openrouter, …) instead of locally. The request still goes through
    /// `llmman serve`, which forwards it upstream; MODEL is the
    /// provider's own model id. See `llmman providers`.
    #[arg(long, short = 'p', value_name = "PROVIDER")]
    pub provider: Option<String>,
    /// Forwarded as Ollama's own top-level `think` field on every request
    /// this sends (see cmd::serve's think_to_chat_template_kwargs) —
    /// `--think false` disables a reasoning model's thinking block
    /// entirely, `--think true` forces it on. Omitted (leaving the
    /// model's own template default in effect) if not passed at all.
    #[arg(long)]
    pub think: Option<bool>,
    /// Forwarded as `options.num_predict` (Ollama's own name for
    /// llama-server's `max_tokens` — see opt_u32 in cmd::serve) on every
    /// request this sends: a hard ceiling on how many tokens a single
    /// reply may generate, regardless of *why* it might otherwise run
    /// away (a real, no-stopping-condition-hit degenerate loop, observed
    /// directly with qwen3.5:0.8b even with `--think false` and a
    /// repeat_penalty already in effect — see this repo's own git history
    /// — is not reliably preventable any other way). Omitted (no ceiling
    /// at all, matching Ollama's own num_predict default of -1) if not
    /// passed.
    #[arg(long)]
    pub num_predict: Option<u32>,
    /// Mirrors ollama's own `run --nowordwrap` — opts out of the
    /// terminal-width word wrap in `wrap_write` below.
    #[arg(long)]
    pub nowordwrap: bool,
    #[arg(
        value_name = "PROMPT",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub prompt: Vec<String>,
}

/// Per-request knobs threaded through unchanged from `RunArgs`.
#[derive(Debug, Clone, Copy)]
struct ChatOptions<'a> {
    think: Option<bool>,
    num_predict: Option<u32>,
    /// Inverse of `RunArgs::nowordwrap`, matching ollama's
    /// `runOptions.WordWrap` (defaults `true`).
    word_wrap: bool,
    /// Provider key to send with every request under `--provider` (see
    /// `provider_model`), `None` for a local model. Applied as a default
    /// header, not a body field — see `chat_client`.
    api_key: Option<&'a str>,
}

impl Default for ChatOptions<'_> {
    fn default() -> Self {
        Self {
            think: None,
            num_predict: None,
            word_wrap: true,
            api_key: None,
        }
    }
}

pub fn run(args: &RunArgs) -> anyhow::Result<()> {
    let provider = crate::providers::provider_flag(args.provider.as_deref())?;
    let prompt = args.prompt.join(" ");

    // resolve_ollama_api, not resolve: `llmman run` is an /api/chat client
    // (see chat_submit/run_interactive_tty below), so a bare name must
    // resolve the same way it would if requested directly over the Ollama
    // API — otherwise a name resolved here, then handed to every /api/chat
    // request this sends, is no longer "bare" by the time ensure_model
    // resolves it server-side (it already has a "/" and a "."), so the
    // docker.io/ai/ default never fires and this silently falls back to
    // hf.co/<name> instead.
    //
    // Not applied under `--provider`: that MODEL names a model on someone
    // else's servers, so no shortname aliasing, tag defaulting, store
    // lookup or pull applies — the same call ensure_model makes
    // server-side. Resolved before the daemon starts, so a malformed
    // local reference still fails without one.
    let route = match provider {
        Some(provider) => Route::Provider(provider),
        None => Route::Local(crate::shortnames::resolve_ollama_api(&args.model)?),
    };

    // Starts `llmman serve` detached, left running indefinitely, if one
    // isn't already reachable — the same shared helper pull/push/launch
    // use (see daemon::ensure_server's doc comment for why stdio is
    // redirected there: without it, this command would hang forever
    // waiting for the (never-exiting) daemon's inherited stdout/stderr
    // pipes to close). No preload model is passed: the resulting daemon
    // is a plain `llmman serve` with no model argument, so it's shared
    // cleanly across every future `run`/`pull`/`push`/`launch` in this
    // session rather than looking like it's dedicated to whatever model
    // happened to start it first.
    //
    // A provider-routed model has nothing to preload or pull, but needs
    // the daemon sooner still: it owns the catalog `provider_model`
    // validates against, and it forwards the chat upstream.
    crate::daemon::ensure_server("")?;

    let (model, api_key) = match route {
        Route::Provider(provider) => provider_model(provider, &args.model)?,
        Route::Local(model) => {
            // Fail fast on a bad/unresolvable reference — mirrors ollama's
            // RunHandler, which resolves (Show, falling back to Pull) the
            // model before ever showing its interactive prompt. Without
            // this, an error like an invalid `hf.co/...` reference wouldn't
            // surface until the first message was submitted to /api/chat,
            // well after the `> ` prompt had already been shown and read
            // from.
            crate::daemon::ensure_model_pulled(&model)?;
            (model, None)
        }
    };

    // Mirrors ollama's RunHandler: interactive needs *both* ends of the
    // terminal, not just stdin — otherwise a redirected stdout still gets
    // raw-moded and starts emitting ANSI escapes into it.
    let interactive = prompt.is_empty() && io::stdin().is_terminal() && io::stdout().is_terminal();
    let opts = ChatOptions {
        think: args.think,
        num_predict: args.num_predict,
        word_wrap: !args.nowordwrap,
        api_key: api_key.as_deref(),
    };

    if interactive {
        run_interactive_tty(&model, opts)
    } else {
        let p = if prompt.is_empty() {
            let mut s = String::new();
            io::stdin().read_line(&mut s)?;
            s.trim().to_string()
        } else {
            prompt
        };
        if !p.is_empty() {
            // One-shot "conversation" over /api/chat, same helper as
            // interactive mode. A fresh tokio runtime, not
            // reqwest::blocking, so Ctrl-C can race the response via
            // `tokio::select!` in `chat_submit` — see that fn's doc comment.
            let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
            let client = chat_client(opts.api_key)?;
            chat_submit(&rt, &client, &model, &mut Vec::new(), p, opts)?;
        }
        Ok(())
    }
}

/// Where this chat goes: a local reference, already resolved (see `run`),
/// or a provider id `provider_model` still has to validate.
enum Route<'a> {
    Local(String),
    Provider(&'a str),
}

/// Validates `--provider`/MODEL against the daemon's catalog, returning
/// the reference the daemon routes on (see
/// [`crate::providers::REMOTE_PREFIX`]) and the key to send with each
/// request, if this shell has one.
///
/// Mirrors `cmd::launch`'s `resolve_provider_model` minus the
/// integrations: the key travels per request in an `Authorization` header
/// (see `client_api_key` in cmd::serve), never to disk.
fn provider_model(provider: &str, model: &str) -> anyhow::Result<(String, Option<String>)> {
    // Same rule as `launch --provider` (see check_provider_supported in
    // cmd::launch): the daemon has no TLS, so a key sent to one elsewhere
    // on the network would cross it in cleartext. A wildcard bind is
    // fine — that hop is still loopback.
    anyhow::ensure!(
        crate::daemon::connects_over_loopback(),
        "--provider needs a local llmman serve: LLMMAN_HOST points at {}, and the provider \
         key would cross the network in cleartext.\n\
         Export the key where that daemon runs, and run llmman there.",
        crate::daemon::server()
    );

    let entry = daemon::provider(provider)?;
    let model = model.trim();
    anyhow::ensure!(
        !model.is_empty(),
        "--provider {provider} also needs a model\n\n{}",
        crate::providers::example_models(&entry.name, &entry.model_ids())
    );

    entry.warn_unlisted(model);

    // Naming the missing variable beats a 401 mid-conversation — unless
    // the daemon has the key, in which case it spends its own.
    let key = entry.api_key();
    anyhow::ensure!(
        key.is_some() || entry.daemon_key_usable(),
        "no API key for {} — set {} in your environment",
        entry.name,
        entry.key_env
    );
    Ok((crate::providers::format_remote_ref(provider, model), key))
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Msg {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
}

/// Async, not `reqwest::blocking`, so a chat turn's response can be raced
/// against `tokio::signal::ctrl_c()` in `chat_submit` — no `.timeout()`
/// needed either, unlike the blocking client's own 30s default.
///
/// A `--provider` key (see `provider_model`) rides along as a default
/// `Authorization` header — this client has one destination, and that
/// header is what `client_api_key` in cmd::serve reads. Sensitive, so a
/// `Debug`-formatted client or request cannot print it.
fn chat_client(api_key: Option<&str>) -> anyhow::Result<Client> {
    let mut builder = Client::builder();
    if let Some(key) = api_key {
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .context("provider API key is not a valid HTTP header value")?;
        value.set_sensitive(true);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, value);
        builder = builder.default_headers(headers);
    }
    builder.build().context("build http client")
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: &'a [Msg],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatReqOptions>,
}

#[derive(Serialize)]
struct ChatReqOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<Msg>,
    #[serde(default)]
    done: bool,
}

// ---------------------------------------------------------------------------
// Terminal rendering — ported from ollama's cmd.go (displayResponse,
// thinkingOutputOpeningText/ClosingText): the "Thinking..." block is the
// one place `ollama run` colors anything.
// ---------------------------------------------------------------------------

/// ollama's readline.ColorGrey/ColorBold/ColorDefault (readline/types.go).
const COLOR_GREY: &str = "\x1b[38;5;245m";
const COLOR_BOLD: &str = "\x1b[1m";
const COLOR_DEFAULT: &str = "\x1b[0m";

/// Mirrors ollama's `thinkingOutputOpeningText`. Ends re-applying grey
/// (not a full reset) so the streamed thinking text after it can rely on
/// that still-active SGR state instead of coloring itself.
fn thinking_opening_text(plain: bool) -> String {
    let text = "Thinking...\n";
    if plain {
        text.to_string()
    } else {
        format!("{COLOR_GREY}{COLOR_BOLD}{text}{COLOR_DEFAULT}{COLOR_GREY}")
    }
}

/// Mirrors ollama's `thinkingOutputClosingText` — a full reset this time.
fn thinking_closing_text(plain: bool) -> String {
    let text = "...done thinking.\n\n";
    if plain {
        text.to_string()
    } else {
        format!("{COLOR_GREY}{COLOR_BOLD}{text}{COLOR_DEFAULT}")
    }
}

/// Mirrors ollama's `displayResponseState`.
#[derive(Default)]
struct WrapState {
    line_length: usize,
    word_buffer: String,
}

/// Terminal width, falling back to 80 like ollama's `displayResponse`
/// does when `term.GetSize` fails (e.g. not a real terminal).
fn term_width() -> usize {
    #[cfg(unix)]
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
            return ws.ws_col as usize;
        }
    }
    80
}

/// Pure computation half of `wrap_write`, split out so it's testable
/// without a real terminal. Direct port of ollama's `displayResponse`:
/// track the current word, and once a line would overflow, backtrack to
/// its start, clear to end of line, and continue on the next one. Every
/// char counts as one column (no `runewidth`-style CJK width handling).
fn wrap_chunk(content: &str, wrap: bool, width: usize, state: &mut WrapState) -> String {
    let mut out = String::new();
    if wrap && width >= 10 {
        for ch in content.chars() {
            if state.line_length + 1 > width - 5 {
                if state.word_buffer.chars().count() > width - 10 {
                    out.push_str(&state.word_buffer);
                    out.push(ch);
                    state.word_buffer.clear();
                    state.line_length = 0;
                    continue;
                }
                let a = state.word_buffer.chars().count();
                if a > 0 {
                    out.push_str(&format!("\x1b[{a}D"));
                }
                out.push_str("\x1b[K\n");
                out.push_str(&state.word_buffer);
                out.push(ch);
                state.line_length = state.word_buffer.chars().count() + 1;
            } else {
                out.push(ch);
                state.line_length += 1;
                match ch {
                    ' ' | '\t' => state.word_buffer.clear(),
                    '\n' | '\r' => {
                        state.line_length = 0;
                        state.word_buffer.clear();
                    }
                    _ => state.word_buffer.push(ch),
                }
            }
        }
    } else {
        out.push_str(&state.word_buffer);
        out.push_str(content);
        state.word_buffer.clear();
    }
    out
}

/// Streams `content` to stdout, word-wrapped via `wrap_chunk` above.
fn wrap_write(content: &str, wrap: bool, state: &mut WrapState) {
    let out = wrap_chunk(content, wrap, term_width(), state);
    print!("{out}");
    io::stdout().flush().ok();
}

/// Mirrors ollama's `progress.NewSpinner("")` (`chat`/`generate` in
/// cmd.go): a braille spinner shown until the first streamed response
/// object arrives, same glyphs and 100ms tick. Unlike ollama, only shown
/// when stderr is a real terminal, so a piped stderr doesn't get raw
/// escape sequences dumped into it.
fn start_spinner() -> Option<ProgressBar> {
    if !io::stderr().is_terminal() {
        return None;
    }
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner} ")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    Some(pb)
}

/// RAII wrapper around `start_spinner`'s result — mirrors ollama's own
/// `defer p.StopAndClear()`: whichever of `chat_submit_async`'s
/// early-return paths fires, this clears any still-running spinner
/// exactly once.
struct SpinnerGuard(Option<ProgressBar>);

impl SpinnerGuard {
    /// Called on Ctrl-C or once the first response object is decoded —
    /// a no-op after the first call, like ollama's own `Spinner.Stop`.
    fn stop(&mut self) {
        if let Some(sp) = self.0.take() {
            sp.finish_and_clear();
        }
    }
}

impl Drop for SpinnerGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// Interactive — TTY path
// ---------------------------------------------------------------------------

fn run_interactive_tty(model: &str, opts: ChatOptions<'_>) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        run_interactive_unix(model, opts)
    }
    #[cfg(not(unix))]
    {
        // Windows fallback: basic cooked-mode loop
        run_interactive_cooked(model, opts)
    }
}

// ---------------------------------------------------------------------------
// Interactive — Unix raw-mode readline (ported from ollama readline package)
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn run_interactive_unix(model: &str, opts: ChatOptions<'_>) -> anyhow::Result<()> {
    use unix_readline::Readline;

    // One tokio runtime, reused across every turn's `block_on` — see
    // `chat_submit`'s doc comment for why this needs async reqwest.
    let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let client = chat_client(opts.api_key)?;
    let mut messages: Vec<Msg> = Vec::new();
    let mut rl = Readline::new()?;
    let mut multiline: Option<String> = None; // Some while inside """
                                              // Accumulates lines while rl.pasting, mirrors ollama's `sb`.
    let mut paste_sb = String::new();

    loop {
        // ". " is ollama's AltPrompt: shown both inside a """ block
        // and while a bracketed paste is still accumulating.
        let prompt = if multiline.is_some() || !paste_sb.is_empty() {
            ". "
        } else {
            "> "
        };

        let line = match rl.readline(prompt) {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(unix_readline::ReadlineError::Interrupted) => {
                multiline = None;
                paste_sb.clear();
                continue;
            }
        };

        // ── Bracketed paste accumulation ────────────────────────────────────
        // Mirrors `case scanner.Pasting: fmt.Fprintln(&sb, line); continue`
        // rl.pasting is true while between \x1b[200~ and \x1b[201~.
        // While pasting, ACCUMULATE into paste_sb WITHOUT submitting.
        // When pasting ends, the final line falls through to normal handling
        // with paste_sb prepended — same as ollama's `default: sb.WriteString`.
        if rl.pasting {
            paste_sb.push_str(&line);
            paste_sb.push('\n');
            continue;
        }

        // Not pasting: prepend any accumulated paste content to this line.
        // (ollama: `default: sb.WriteString(line)` then submit if sb.Len()>0)
        let line = if !paste_sb.is_empty() {
            let mut full = std::mem::take(&mut paste_sb);
            full.push_str(&line);
            full
        } else {
            line
        };

        // ── """ multiline mode ───────────────────────────────────────────────
        if let Some(ref mut buf) = multiline {
            if let Some(content) = line.strip_suffix("\"\"\"") {
                buf.push_str(content);
                let full = std::mem::take(buf).trim_end_matches('\n').to_string();
                multiline = None;
                if !full.trim().is_empty() {
                    submit_turn(&rt, &client, model, &mut rl, &mut messages, full, opts)?;
                }
            } else {
                buf.push_str(&line);
                buf.push('\n');
            }
            continue;
        }

        // ── Slash commands ───────────────────────────────────────────────────
        match line.trim() {
            "" => continue,
            "/bye" | "/exit" => break,
            "/clear" => {
                messages.clear();
                eprintln!("Conversation cleared.");
                continue;
            }
            s if s.starts_with('/') => {
                eprintln!("Commands: /bye  /clear  \"\"\" (multiline)");
                continue;
            }
            _ => {}
        }

        // ── Triple-quote multiline opener ────────────────────────────────────
        if line.trim_start().starts_with("\"\"\"") {
            let inner = line.trim_start().trim_start_matches("\"\"\"");
            if let Some(closed) = inner.strip_suffix("\"\"\"") {
                let content = closed.to_string();
                if !content.trim().is_empty() {
                    submit_turn(&rt, &client, model, &mut rl, &mut messages, content, opts)?;
                }
            } else {
                multiline = Some(inner.to_string() + "\n");
            }
            continue;
        }

        if !line.trim().is_empty() {
            submit_turn(&rt, &client, model, &mut rl, &mut messages, line, opts)?;
        }
    }

    Ok(())
}

/// Wraps `chat_submit` with the raw-mode toggle that lets Ctrl-C actually
/// interrupt a streaming response — mirrors ollama's
/// `readline.Instance.Readline()`, which holds raw mode only around
/// reading one line. Without this, raw mode (ISIG off) would stay active
/// through the response too, so Ctrl-C would just sit as an unread byte 3
/// until the next `rl.readline()` call instead of interrupting now.
#[cfg(unix)]
fn submit_turn(
    rt: &tokio::runtime::Runtime,
    client: &Client,
    model: &str,
    rl: &mut unix_readline::Readline,
    messages: &mut Vec<Msg>,
    content: String,
    opts: ChatOptions<'_>,
) -> anyhow::Result<()> {
    rl.leave_raw();
    let result = chat_submit(rt, client, model, messages, content, opts);
    rl.enter_raw();
    result
}

/// Sends one chat turn and streams the response, racing it against
/// Ctrl-C — mirrors ollama's per-turn `context.WithCancel` +
/// `signal.Notify(SIGINT)`: Ctrl-C stops just this turn's response
/// without killing the process. An interrupted turn's partial reply is
/// *not* added to `messages`, matching cmd.go's `chat` returning `nil`
/// on `context.Canceled`.
///
/// Async, not `reqwest::blocking`, so the stream can be raced against
/// `tokio::signal::ctrl_c()` via `tokio::select!` below.
fn chat_submit(
    rt: &tokio::runtime::Runtime,
    client: &Client,
    model: &str,
    messages: &mut Vec<Msg>,
    content: String,
    opts: ChatOptions<'_>,
) -> anyhow::Result<()> {
    rt.block_on(chat_submit_async(client, model, messages, content, opts))
}

async fn chat_submit_async(
    client: &Client,
    model: &str,
    messages: &mut Vec<Msg>,
    content: String,
    opts: ChatOptions<'_>,
) -> anyhow::Result<()> {
    messages.push(Msg {
        role: "user".into(),
        content,
        thinking: None,
    });

    // Mirrors ollama's `progress.NewSpinner("")`: ticks on stderr until
    // the first token/thinking chunk arrives, so `llmman run` doesn't sit
    // silently while a cold model loads server-side.
    let mut spinner = SpinnerGuard(start_spinner());

    // One listener for this whole turn, mirroring ollama's own
    // `signal.Notify(sigChan, syscall.SIGINT)` called once per turn (not
    // per streamed chunk). Recreating `tokio::signal::ctrl_c()` on every
    // loop iteration would subscribe a fresh listener each time, and a
    // SIGINT delivered in the gap between the old one dropping and the
    // new one subscribing could be missed — pinning it once and reusing
    // it via `&mut` closes that gap.
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    let send = client
        .post(format!("{}/api/chat", daemon::server()))
        .json(&ChatReq {
            model,
            messages,
            stream: true,
            think: opts.think,
            options: opts.num_predict.map(|n| ChatReqOptions {
                num_predict: Some(n),
            }),
        })
        .send();

    let resp = tokio::select! {
        r = send => r.context("connect to llmman serve")?,
        // Nothing printed yet, so just stop — `spinner`'s Drop clears it.
        _ = &mut ctrl_c => return Ok(()),
    };

    if !resp.status().is_success() {
        spinner.stop();
        let body = resp.text().await.unwrap_or_default();
        // The daemon's own message, not the JSON envelope around it: a
        // provider's 401 naming the key it rejected is the whole of what
        // is needed here.
        anyhow::bail!("{}", daemon::api_error(&body).unwrap_or(body));
    }

    // Stream NDJSON lines as they arrive, async so each read can be
    // `tokio::select!`ed against Ctrl-C below.
    let byte_stream = resp.bytes_stream().map_err(io::Error::other);
    let mut lines =
        tokio::io::BufReader::new(tokio_util::io::StreamReader::new(byte_stream)).lines();

    // Mirrors ollama's `plainText`: no ANSI codes when stdout is redirected.
    let plain_text = !io::stdout().is_terminal();
    let mut full = String::new();
    let mut thinking_content = String::new();
    let mut thinking_open = false;
    let mut thinking_closed = false;
    let mut wrap = WrapState::default();
    let mut interrupted = false;

    loop {
        let line = tokio::select! {
            l = lines.next_line() => l.context("read response stream")?,
            // Same pinned listener as above, see its comment for why.
            _ = &mut ctrl_c => { interrupted = true; break; }
        };
        let Some(line) = line else { break };
        if line.is_empty() {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<ChatChunk>(&line) else {
            continue;
        };
        // Cleared on the first decoded response object, like ollama's
        // own `fn` callback.
        spinner.stop();
        if let Some(ref msg) = chunk.message {
            if let Some(ref t) = msg.thinking {
                if !t.is_empty() {
                    if !thinking_open {
                        print!("{}", thinking_opening_text(plain_text));
                        thinking_open = true;
                        thinking_closed = false;
                    }
                    thinking_content.push_str(t);
                    wrap_write(t, opts.word_wrap, &mut wrap);
                }
            }
            if thinking_open && !thinking_closed && !msg.content.is_empty() {
                if !thinking_content.ends_with('\n') {
                    println!();
                }
                print!("{}", thinking_closing_text(plain_text));
                thinking_open = false;
                thinking_closed = true;
                wrap = WrapState::default();
            }
            if !msg.content.is_empty() {
                wrap_write(&msg.content, opts.word_wrap, &mut wrap);
                full.push_str(&msg.content);
            }
        }
        if chunk.done {
            break;
        }
    }

    if interrupted {
        return Ok(());
    }

    println!("\n");
    messages.push(Msg {
        role: "assistant".into(),
        content: full,
        thinking: None,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Unix raw-mode readline — direct port of ollama readline/readline.go
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod unix_readline {
    use std::io::{BufReader, Read, Stdin, Write};
    use std::os::unix::io::AsRawFd;

    // Character codes — identical to ollama readline/types.go
    const CHAR_INTERRUPT: u8 = 3; // Ctrl-C
    const CHAR_EOF: u8 = 4; // Ctrl-D
    const CHAR_CTRL_J: u8 = 10; // \n  line feed / pasted newline
    const CHAR_ENTER: u8 = 13; // \r  keyboard Enter
    const CHAR_ESC: u8 = 27;
    const CHAR_ESCAPE_EX: u8 = 91; // '[' — second byte of ESC[
    const CHAR_BACKSPACE: u8 = 127;

    pub enum ReadlineError {
        Interrupted,
    }

    // CharBracketedPaste = 50 ('2') — third byte of ESC[ sequence;
    // reading 3 more bytes gives "00~" (paste start) or "01~" (paste end).
    // Mirrors ollama readline/types.go: CharBracketedPaste/Start/End.
    const CHAR_BRACKETED_PASTE: u8 = 50; // '2'
    const PASTE_START: &[u8; 3] = b"00~";
    const PASTE_END: &[u8; 3] = b"01~";

    pub struct Readline {
        reader: BufReader<Stdin>,
        orig: libc::termios,
        fd: std::os::unix::io::RawFd,
        pub pasting: bool, // true while inside \x1b[200~...\x1b[201~
    }

    impl Readline {
        /// Enable raw mode + bracketed paste (mirrors ollama SetRawMode + StartBracketedPaste).
        pub fn new() -> anyhow::Result<Self> {
            let stdin = std::io::stdin();
            let fd = stdin.as_raw_fd();

            let orig = unsafe {
                let mut t: libc::termios = std::mem::zeroed();
                if libc::tcgetattr(fd, &mut t) < 0 {
                    anyhow::bail!("tcgetattr failed");
                }
                t
            };

            let rl = Self {
                reader: BufReader::new(stdin),
                orig,
                fd,
                pasting: false,
            };
            if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &rl.raw_termios()) } < 0 {
                anyhow::bail!("tcsetattr failed");
            }

            // Enable bracketed paste for the whole session (toggled off
            // once, in Drop) — mirrors ollama's own start/end pair
            // wrapping the entire REPL loop, not each `Readline()` call.
            print!("\x1b[?2004h");
            std::io::stdout().flush().ok();

            Ok(rl)
        }

        /// Raw-mode termios derived from `orig`, mirrors `SetRawMode`.
        fn raw_termios(&self) -> libc::termios {
            let mut raw = self.orig;
            raw.c_iflag &= !(libc::IGNBRK
                | libc::BRKINT
                | libc::PARMRK
                | libc::ISTRIP
                | libc::INLCR
                | libc::IGNCR
                | libc::ICRNL
                | libc::IXON);
            raw.c_lflag &= !(libc::ECHO | libc::ECHONL | libc::ICANON | libc::ISIG | libc::IEXTEN);
            raw.c_cflag &= !(libc::CSIZE | libc::PARENB);
            raw.c_cflag |= libc::CS8;
            raw.c_cc[libc::VMIN] = 1;
            raw.c_cc[libc::VTIME] = 0;
            raw
        }

        /// (Re-)enters raw mode — used by `new()` and by `run::submit_turn`
        /// to undo `leave_raw` before the next `readline()` call.
        pub fn enter_raw(&self) {
            unsafe {
                libc::tcsetattr(self.fd, libc::TCSANOW, &self.raw_termios());
            }
        }

        /// Restores cooked (ISIG-enabled) mode for the duration of a
        /// streamed response, so a real SIGINT reaches `chat_submit`'s
        /// `ctrl_c()` instead of raw mode swallowing it as a plain byte 3.
        pub fn leave_raw(&self) {
            unsafe {
                libc::tcsetattr(self.fd, libc::TCSANOW, &self.orig);
            }
        }

        /// Read one logical line from the terminal.
        ///
        /// Paste detection mirrors ollama readline/readline.go exactly:
        ///   - After each read, check reader.buffer() (≡ reader.Buffered() in Go)
        ///   - If non-empty → draining (we are consuming a paste)
        ///   - CharCtrlJ (\n) while draining → submit (same as Enter)
        ///   - CharCtrlJ while NOT draining → Ctrl-J multiline continuation
        ///   - CharEnter (\r) → always submit
        pub fn readline(&mut self, prompt: &str) -> Result<Option<String>, ReadlineError> {
            print!("{prompt}");
            std::io::stdout().flush().ok();

            let mut buf: Vec<u8> = Vec::new();
            let mut pasted_lines: Vec<String> = Vec::new();
            let mut draining = false;
            let mut stop_draining = false;
            let mut esc = false;
            let mut esc_ex = false;

            loop {
                // Apply deferred state from previous iteration (ollama lines 130-134)
                if stop_draining {
                    draining = false;
                    stop_draining = false;
                }

                // Read exactly one byte
                let mut b = [0u8; 1];
                match self.reader.read_exact(&mut b) {
                    Ok(_) => {}
                    Err(_) => return Ok(None),
                }
                let r = b[0];

                // Paste detection: mirrors `if i.Terminal.reader.Buffered() > 0`
                if !self.reader.buffer().is_empty() {
                    draining = true;
                } else if draining {
                    stop_draining = true;
                }

                // ESC sequence handling — mirrors ollama readline.go escex block.
                // Key addition: CharBracketedPaste ('2') reads 3 more bytes to
                // detect "00~" (paste start) or "01~" (paste end).
                if esc_ex {
                    esc_ex = false;
                    match r {
                        CHAR_BRACKETED_PASTE => {
                            // Read 3 more bytes: "00~" or "01~"
                            let mut code = [0u8; 3];
                            if self.reader.read_exact(&mut code).is_ok() {
                                if &code == PASTE_START {
                                    self.pasting = true;
                                } else if &code == PASTE_END {
                                    self.pasting = false;
                                }
                                // Update draining after reading extra bytes
                                if !self.reader.buffer().is_empty() {
                                    draining = true;
                                }
                            }
                        }
                        // Consume the '~' for delete/other 2-byte sequences
                        51 | 53 | 54 => {
                            let mut tilde = [0u8; 1];
                            let _ = self.reader.read_exact(&mut tilde);
                        }
                        _ => {} // arrow keys etc. — just skip
                    }
                    continue;
                } else if esc {
                    esc = false;
                    if r == CHAR_ESCAPE_EX {
                        esc_ex = true;
                    }
                    continue;
                }

                match r {
                    CHAR_INTERRUPT => {
                        pasted_lines.clear();
                        buf.clear();
                        println!();
                        return Err(ReadlineError::Interrupted);
                    }
                    CHAR_EOF => {
                        // Mirrors ollama's `case CharDelete`: only checks
                        // the current line's own buffer, so Ctrl-D on a
                        // fresh empty continuation line exits the whole
                        // REPL even mid-multiline-entry.
                        if buf.is_empty() {
                            println!();
                            return Ok(None);
                        }
                    }
                    CHAR_ESC => {
                        esc = true;
                    }
                    CHAR_BACKSPACE => {
                        if !buf.is_empty() {
                            // Remove last complete UTF-8 codepoint
                            loop {
                                match buf.pop() {
                                    None => break,
                                    Some(b) if (b & 0xC0) != 0x80 => break, // lead byte
                                    Some(_) => {} // continuation byte, keep going
                                }
                            }
                            print!("\x08 \x08");
                            std::io::stdout().flush().ok();
                        } else if !pasted_lines.is_empty() {
                            let prev = pasted_lines.pop().unwrap();
                            print!("\r\x1b[K\x1b[A\r\x1b[K{prompt}{prev}");
                            std::io::stdout().flush().ok();
                            buf = prev.into_bytes();
                        }
                    }
                    CHAR_CTRL_J => {
                        // \n: pasted newline (draining) or Ctrl-J multiline (not draining)
                        // Mirrors ollama case CharCtrlJ
                        if !draining {
                            // Not draining → multiline continuation (Ctrl-J typed)
                            pasted_lines.push(String::from_utf8_lossy(&buf).to_string());
                            buf.clear();
                            println!();
                            print!(". ");
                            std::io::stdout().flush().ok();
                        } else {
                            // Draining → submit (pasted \n acts like Enter)
                            return Ok(Some(Self::assemble(&mut buf, &mut pasted_lines)));
                        }
                    }
                    CHAR_ENTER => {
                        // \r: keyboard Enter → always submit
                        return Ok(Some(Self::assemble(&mut buf, &mut pasted_lines)));
                    }
                    c => {
                        // Printable ASCII, tab, or UTF-8 bytes
                        if c >= 32 || c == 9 || c >= 0x80 {
                            buf.push(c);
                            let _ = std::io::stdout().write_all(&[c]);
                            std::io::stdout().flush().ok();
                        }
                    }
                }
            }
        }

        fn assemble(buf: &mut Vec<u8>, pasted_lines: &mut Vec<String>) -> String {
            let last = String::from_utf8_lossy(buf).to_string();
            buf.clear();
            println!();
            if pasted_lines.is_empty() {
                last
            } else {
                let prefix = pasted_lines.join("\n");
                pasted_lines.clear();
                format!("{prefix}\n{last}")
            }
        }
    }

    impl Drop for Readline {
        fn drop(&mut self) {
            // Disable bracketed paste, restore terminal — mirrors ollama's defer
            print!("\x1b[?2004l");
            std::io::stdout().flush().ok();
            unsafe {
                libc::tcsetattr(self.fd, libc::TCSANOW, &self.orig);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Windows / non-TTY fallback (cooked mode)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn run_interactive_cooked(model: &str, opts: ChatOptions<'_>) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let client = chat_client(opts.api_key)?;
    let mut messages: Vec<Msg> = Vec::new();
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut reader = std::io::BufReader::new(stdin.lock());

    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line
            .trim_end_matches('\n')
            .trim_end_matches('\r')
            .to_string();
        match line.trim() {
            "" => continue,
            "/bye" | "/exit" => break,
            "/clear" => {
                messages.clear();
                continue;
            }
            _ => {}
        }
        if !line.trim().is_empty() {
            chat_submit(&rt, &client, model, &mut messages, line, opts)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_options_default_matches_ollamas_own_wordwrap_true_default() {
        let opts = ChatOptions::default();
        assert_eq!(opts.think, None);
        assert_eq!(opts.num_predict, None);
        assert!(opts.word_wrap);
    }

    #[test]
    fn thinking_opening_text_plain_has_no_ansi_codes() {
        assert_eq!(thinking_opening_text(true), "Thinking...\n");
    }

    #[test]
    fn thinking_opening_text_colored_wraps_in_grey_bold_then_grey_again() {
        let text = thinking_opening_text(false);
        assert!(text.starts_with(&format!("{COLOR_GREY}{COLOR_BOLD}")));
        assert!(text.contains("Thinking...\n"));
        // Re-applies grey rather than a full reset.
        assert!(text.ends_with(&format!("{COLOR_DEFAULT}{COLOR_GREY}")));
    }

    #[test]
    fn thinking_closing_text_plain_has_no_ansi_codes() {
        assert_eq!(thinking_closing_text(true), "...done thinking.\n\n");
    }

    #[test]
    fn thinking_closing_text_colored_ends_in_a_full_reset() {
        let text = thinking_closing_text(false);
        assert!(text.starts_with(&format!("{COLOR_GREY}{COLOR_BOLD}")));
        assert!(text.contains("...done thinking.\n\n"));
        assert!(text.ends_with(COLOR_DEFAULT));
        assert!(!text.ends_with(&format!("{COLOR_DEFAULT}{COLOR_GREY}")));
    }

    #[test]
    fn wrap_chunk_disabled_passes_content_through_unchanged() {
        let mut state = WrapState::default();
        let out = wrap_chunk(
            "hello there, this line is not wrapped at all",
            false,
            20,
            &mut state,
        );
        assert_eq!(out, "hello there, this line is not wrapped at all");
    }

    #[test]
    fn wrap_chunk_too_narrow_terminal_passes_content_through_unchanged() {
        // Mirrors ollama's `wordWrap && termWidth >= 10` guard.
        let mut state = WrapState::default();
        let out = wrap_chunk("hello there", true, 9, &mut state);
        assert_eq!(out, "hello there");
    }

    #[test]
    fn wrap_chunk_breaks_before_a_word_that_would_overflow_the_line() {
        let mut state = WrapState::default();
        // width=20: overflow threshold is line_length+1 > width-5 (>15).
        let out = wrap_chunk("one two three four five", true, 20, &mut state);
        // "four" gets backtracked onto a new line instead of splitting.
        assert!(out.contains("\x1b[K\n"));
        let after_break = out.split("\x1b[K\n").nth(1).unwrap();
        assert!(after_break.starts_with("four"));
    }

    #[test]
    fn wrap_chunk_state_persists_across_calls_like_a_single_stream() {
        // Several small calls (one per streamed token) must wrap the
        // same as one big call, carried via WrapState between them.
        let mut state_streamed = WrapState::default();
        let mut streamed = String::new();
        for word in ["one ", "two ", "three ", "four ", "five"] {
            streamed.push_str(&wrap_chunk(word, true, 20, &mut state_streamed));
        }

        let mut state_whole = WrapState::default();
        let whole = wrap_chunk("one two three four five", true, 20, &mut state_whole);

        assert_eq!(streamed, whole);
    }

    #[test]
    fn wrap_chunk_an_unbroken_word_longer_than_the_line_flushes_without_backtracking() {
        // A word too long to ever fit its own line skips the backtrack
        // and just flushes the buffered prefix again — faithfully
        // mirroring ollama's own `fmt.Printf("%s%c", wordBuffer, ch)`
        // here, the buffered prefix really does appear twice in the
        // output for this rare edge case.
        let mut state = WrapState::default();
        let out = wrap_chunk("supercalifragilisticexpialidocious", true, 20, &mut state);
        assert_eq!(
            out,
            "supercalifragilsupercalifragilisticexpialidocisticexpialidocious"
        );
    }
}
