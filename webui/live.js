// llmman live — a hands-free voice + video conversation UI over the
// daemon's own /llmman/live/turn endpoint.
//
// Hand-written ES module served verbatim (gzipped) by `handle_live_js`;
// unlike webui/bundle.js this file is source, not a build artifact, so
// edit it directly.
//
// The shape of a turn:
//
//   microphone → the browser's own speech recognizer → a sentence
//   + the last N JPEG frames from the camera → POST /llmman/live/turn
//   → SSE back → the reply, streamed, rendered and spoken.
//
// Speech recognition is the browser's (the Web Speech API), not a model
// llmman loads — see the /llmman/live section of src/cmd/serve.rs for
// why. Recent Chrome can run it entirely on-device; `setUpRecognition`
// below asks for that first and says which one is in use, because "your
// audio left this machine" is not something a local-inference tool should
// let happen quietly.

const TURN_URL = "/llmman/live/turn";
const MODELS_URL = "/v1/models";

// --- turn taking ---------------------------------------------------------

/** Quiet time after the recognizer's last result before the sentences it
 *  produced are sent as one turn. Long enough to keep "Hi. What's on the
 *  table?" together as a single turn rather than firing twice, short
 *  enough not to feel like waiting. */
const COMMIT_DELAY_MS = 700;
/** Floor between recognition restarts. The Web Speech API ends a session
 *  on its own schedule (and immediately, on some errors), so restarting
 *  is normal — but it has to be rate-limited or a permanently failing
 *  recognizer becomes a busy loop. */
const RESTART_DELAY_MS = 250;

// --- video ---------------------------------------------------------------

/** How often a frame is grabbed into the ring buffer. The model only ever
 *  sees the last few, so this is really "how stale the oldest one is". */
const FRAME_INTERVAL_MS = 700;
const FRAME_RING = 8;
/** Must not exceed the daemon's own `MAX_LIVE_FRAMES`. */
const MAX_FRAMES_PER_TURN = 8;
/** Long edge of a captured frame. Vision encoders tile at roughly this
 *  scale anyway, and it keeps a turn's upload in the tens of kilobytes. */
const FRAME_EDGE = 512;
const FRAME_QUALITY = 0.6;

// --- history -------------------------------------------------------------

/** Messages of text-only history replayed to the model. Frames are never
 *  replayed: they are large, and a stale one actively misleads a model
 *  being asked about what the camera sees *now*. */
const HISTORY_MESSAGES = 12;

const $ = (id) => document.getElementById(id);
const el = {
  preview: $("preview"),
  cameraOff: $("camera-off"),
  caption: $("caption"),
  start: $("start"),
  stop: $("stop"),
  mic: $("mic"),
  cam: $("cam"),
  hint: $("hint"),
  model: $("model"),
  system: $("system"),
  sendVideo: $("send-video"),
  speak: $("speak"),
  bargeIn: $("barge-in"),
  frames: $("frames"),
  log: $("log"),
  compose: $("compose"),
  text: $("text"),
};

const session = {
  stream: null,
  micOn: true,
  camOn: true,
  frames: [],
  frameTimer: null,
  inFlight: null,
  history: [],
};

const speech = {
  /** The `SpeechRecognition` constructor, or null where there is none. */
  Recognition: null,
  /** True once the recognizer is confirmed to run without sending audio
   *  anywhere. */
  onDevice: false,
  recognition: null,
  running: false,
  /** Set while deliberately stopping, so `onend` doesn't restart. */
  stopping: false,
  /** Finalized sentences not yet sent as a turn. */
  pending: "",
  commitTimer: null,
  /** When the last turn was handed to `sendTurn` — see the interrupt
   *  guard in `onRecognitionResult`. */
  lastCommitAt: 0,
};

// ---------------------------------------------------------------------------
// Transcript rendering
// ---------------------------------------------------------------------------

function setHint(text, isError) {
  el.hint.textContent = text;
  el.hint.classList.toggle("error", !!isError);
}

function setCaption(text) {
  el.caption.textContent = text;
  el.caption.hidden = !text;
}

function addTurn(kind, who) {
  const wrap = document.createElement("div");
  wrap.className = `turn ${kind}`;
  const label = document.createElement("b");
  label.textContent = who;
  const body = document.createElement("p");
  wrap.append(label, body);
  el.log.append(wrap);
  el.log.scrollTop = el.log.scrollHeight;
  return body;
}

// ---------------------------------------------------------------------------
// Model list
// ---------------------------------------------------------------------------

async function loadModels() {
  let ids = [];
  try {
    const resp = await fetch(MODELS_URL);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    ids = ((await resp.json()).data || []).map((m) => m.id).sort();
  } catch (err) {
    setHint(`Could not list models: ${err.message}`, true);
    return;
  }

  el.model.replaceChildren();
  el.model.append(new Option(ids.length ? "select a model" : "no models", ""));
  for (const id of ids) el.model.append(new Option(id, id));
  if (ids.length) {
    el.model.value = ids[0];
  } else {
    setHint("No models in the store yet — pull one with `llmman pull`.", true);
  }
}

// ---------------------------------------------------------------------------
// Speech recognition (the browser's own)
// ---------------------------------------------------------------------------

const RECOGNITION_LANG = navigator.language || "en-US";

/** Picks the recognizer, preferring one that keeps the audio here.
 *
 *  Chrome 138+ can run recognition on-device and exposes `available()`
 *  to ask whether it currently can; everything else has only the
 *  original API, whose implementation may send audio to a vendor
 *  service. Asking costs one call and is the difference between the
 *  microphone staying on this machine and not. */
async function setUpRecognition() {
  const Recognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    el.mic.hidden = true;
    setHint(
      "This browser has no speech recognition, so llmman live can't listen — " +
        "the camera and the text box below still work. Chrome, Edge and " +
        "Safari can listen.",
      true,
    );
    return;
  }
  speech.Recognition = Recognition;

  // The original API only: no way to ask where the audio goes, and no
  // way to ask for it to stay put either.
  if (typeof Recognition.available !== "function") return;
  const status = await Recognition.available({
    langs: [RECOGNITION_LANG],
    processLocally: true,
  }).catch(() => "unavailable");
  speech.onDevice = status === "available";
}

function startRecognition() {
  if (!speech.Recognition || speech.running || !session.stream) return;
  const recognition = new speech.Recognition();
  recognition.lang = RECOGNITION_LANG;
  // Keep listening across sentences instead of stopping at the first one,
  // and surface partial results so the caption tracks the speaker and
  // barge-in can fire on the first syllable rather than the last.
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  if (speech.onDevice) recognition.processLocally = true;

  recognition.onresult = onRecognitionResult;
  recognition.onerror = onRecognitionError;
  recognition.onend = () => {
    speech.running = false;
    if (!speech.stopping && session.stream && session.micOn) {
      setTimeout(startRecognition, RESTART_DELAY_MS);
    }
  };

  try {
    recognition.start();
    speech.running = true;
    speech.stopping = false;
    speech.recognition = recognition;
  } catch {
    // start() throws if one is somehow already running; onend will
    // schedule the next attempt.
  }
}

function stopRecognition() {
  speech.stopping = true;
  clearTimeout(speech.commitTimer);
  speech.pending = "";
  setCaption("");
  if (speech.recognition) speech.recognition.abort();
  speech.recognition = null;
  speech.running = false;
}

function restartRecognition() {
  stopRecognition();
  speech.stopping = false;
  startRecognition();
}

function onRecognitionResult(event) {
  // Without headphones the recognizer hears the reply being spoken and
  // answers it, which is a conversation with itself. Dropping results
  // while speaking is the only reliable defence, so interrupting is
  // opt-in for people who can't be overheard by their own microphone.
  if (tts.speaking && !el.bargeIn.checked) return;

  let interim = "";
  let final = "";
  for (let i = event.resultIndex; i < event.results.length; i++) {
    const result = event.results[i];
    if (result.isFinal) final += result[0].transcript;
    else interim += result[0].transcript;
  }
  if (!interim && !final) return;

  // The user is talking, so whatever the assistant was saying (or about
  // to say) stops. Matching how a person interrupts is most of what makes
  // this feel live rather than turn-based.
  //
  // Except just after a turn was sent: the recognizer can deliver a
  // trailing result for speech that has *already* been committed, and
  // aborting on that would cancel the very turn it produced. Nothing
  // reaches the speakers that fast, so a window as short as the commit
  // delay separates the two cases.
  if (Date.now() - speech.lastCommitAt > COMMIT_DELAY_MS) interrupt();

  speech.pending = (speech.pending + final).replace(/\s+/g, " ");
  setCaption((speech.pending + " " + interim).trim());

  // Restarted on every result, final or not: the turn is sent once the
  // speaker has actually stopped, not at the first sentence boundary.
  clearTimeout(speech.commitTimer);
  speech.commitTimer = setTimeout(commitSpokenTurn, COMMIT_DELAY_MS);
}

function commitSpokenTurn() {
  const text = speech.pending.trim();
  speech.pending = "";
  setCaption("");
  if (!text) return;
  speech.lastCommitAt = Date.now();
  sendTurn(text);
}

function onRecognitionError(event) {
  switch (event.error) {
    case "no-speech":
    case "aborted":
      return; // ordinary; onend restarts
    case "not-allowed":
    case "service-not-allowed":
      speech.stopping = true;
      setHint(
        "Microphone access was refused, so llmman live can't listen. You can " +
          "still type below.",
        true,
      );
      return;
    case "audio-capture":
      setHint("No microphone was found. You can still type below.", true);
      return;
    case "network":
      setHint(
        "This browser's speech recognition needs a network connection — it " +
          "is not running on this machine. Type below instead.",
        true,
      );
      return;
    default:
      setHint(`Speech recognition failed: ${event.error}`, true);
  }
}

// ---------------------------------------------------------------------------
// Video: ring buffer of recent JPEG stills
// ---------------------------------------------------------------------------

const canvas = document.createElement("canvas");

/** Base64 (no data-URI prefix) of the current video frame, which is the
 *  wire shape /llmman/live/turn wants. */
async function grabFrame() {
  const video = el.preview;
  if (!session.camOn || !video.videoWidth) return;
  const scale = Math.min(
    1,
    FRAME_EDGE / Math.max(video.videoWidth, video.videoHeight),
  );
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  // Drawn from the unmirrored source: the preview is flipped for the
  // person looking at it, but the model should see text the right way
  // round.
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUri = canvas.toDataURL("image/jpeg", FRAME_QUALITY);
  session.frames.push(dataUri.slice(dataUri.indexOf(",") + 1));
  while (session.frames.length > FRAME_RING) session.frames.shift();
}

// ---------------------------------------------------------------------------
// Speech output
// ---------------------------------------------------------------------------

const tts = { queue: "", speaking: false, outstanding: 0 };

/** End of a sentence: terminal punctuation, the closing quote or bracket
 *  that may follow it, then whitespace. Requiring the whitespace is what
 *  keeps "3.5" or "qwen3.5:0.8b" from being read as two sentences. */
const SENTENCE_END = /[.!?…]["')\]]?\s/;

/** Speaks whole sentences as they finish streaming, rather than waiting
 *  for the full reply — the same reason the reply itself is streamed.
 *  `flush` speaks whatever is left at the end of a turn, boundary or
 *  not. */
function speakStreaming(text, flush) {
  if (!el.speak.checked || !window.speechSynthesis) return;
  tts.queue += text;
  for (;;) {
    const end = SENTENCE_END.exec(tts.queue);
    if (!end && !flush) return;
    const cut = end ? end.index + end[0].length : tts.queue.length;
    const say = tts.queue.slice(0, cut).trim();
    tts.queue = tts.queue.slice(cut);
    if (say) speakOne(say);
    if (!tts.queue) return;
  }
}

function speakOne(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  tts.outstanding++;
  tts.speaking = true;
  const finished = () => {
    // `speechSynthesis.speaking` lags its own events, and the recognizer
    // gate above has to be exact — one stale frame of "not speaking" is
    // enough for the assistant to hear itself and reply to it.
    if (--tts.outstanding <= 0) {
      tts.outstanding = 0;
      tts.speaking = false;
    }
  };
  utterance.onend = finished;
  utterance.onerror = finished;
  window.speechSynthesis.speak(utterance);
}

/** Stops the reply, whether it is still being generated or only still
 *  being spoken. */
function interrupt() {
  tts.queue = "";
  tts.outstanding = 0;
  tts.speaking = false;
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  if (session.inFlight) {
    session.inFlight.abort();
    session.inFlight = null;
  }
}

// ---------------------------------------------------------------------------
// One turn: POST the sentence and frames, read the SSE reply
// ---------------------------------------------------------------------------

/** Splits an SSE byte stream into the JSON payload of each `data:` line.
 *  Every event this endpoint sends is a single-line JSON object (see
 *  `LiveEvent` in src/cmd/serve.rs), so no multi-line reassembly is
 *  needed — only buffering across chunk boundaries. */
async function* readEvents(body) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += value;
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trimEnd();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data: ")) continue;
      try {
        yield JSON.parse(line.slice(6));
      } catch {
        // A payload llmman didn't write; nothing useful to do with it.
      }
    }
  }
}

function framesForTurn() {
  // `slice(-0)` is `slice(0)` — the whole buffer — so zero has to be its
  // own case rather than falling out of the arithmetic. Clamped to what
  // the daemon accepts (MAX_LIVE_FRAMES) so a hand-edited input makes a
  // smaller turn rather than a 400.
  if (!el.sendVideo.checked) return [];
  const wanted = Math.max(
    0,
    Math.min(MAX_FRAMES_PER_TURN, Number(el.frames.value) || 0),
  );
  return wanted ? session.frames.slice(-wanted) : [];
}

async function sendTurn(text) {
  const model = el.model.value;
  if (!model) {
    setHint("Pick a model first.", true);
    return;
  }

  interrupt();
  addTurn("user", "you").textContent = text;
  const frames = framesForTurn();

  const controller = new AbortController();
  session.inFlight = controller;

  let replyBody = null;
  let reply = "";

  try {
    const resp = await fetch(TURN_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model,
        text,
        system: el.system.value.trim() || null,
        history: session.history.slice(-HISTORY_MESSAGES),
        frames,
      }),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.error || `${resp.status} ${resp.statusText}`);
    }
    for await (const event of readEvents(resp.body)) {
      switch (event.type) {
        case "context":
          if (!event.vision && frames.length) {
            setHint(
              `${event.model} has no vision support, so the camera frames ` +
                "were not sent. Pick a vision model to be seen.",
            );
          } else if (event.frames) {
            setHint(`Sent ${event.frames} camera frame(s) with this turn.`);
          }
          break;
        case "delta":
          replyBody = replyBody || addTurn("assistant", "llmman");
          reply += event.text;
          replyBody.textContent = reply;
          el.log.scrollTop = el.log.scrollHeight;
          speakStreaming(event.text, false);
          break;
        case "thinking":
          // Reasoning is deliberately neither rendered as the reply nor
          // spoken: it isn't addressed to the user.
          break;
        case "error":
          throw new Error(event.message);
        default:
          break;
      }
    }
    speakStreaming("", true);
    session.history.push({ role: "user", content: text });
    if (reply) session.history.push({ role: "assistant", content: reply });
  } catch (err) {
    if (err.name === "AbortError") return; // interrupted on purpose
    addTurn("failed", "error").textContent = err.message;
    return;
  } finally {
    if (session.inFlight === controller) session.inFlight = null;
  }
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

async function startSession() {
  el.start.disabled = true;
  try {
    // Video only: the recognizer opens the microphone itself, and asking
    // for it here too would capture the same input twice for nothing.
    session.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (err) {
    el.start.disabled = false;
    setHint(
      `Camera access was refused (${err.name}). A browser only grants it ` +
        "over http://127.0.0.1, http://localhost or HTTPS.",
      true,
    );
    return;
  }

  el.preview.srcObject = session.stream;
  el.cameraOff.hidden = true;
  session.frameTimer = setInterval(grabFrame, FRAME_INTERVAL_MS);

  el.start.hidden = true;
  el.stop.hidden = false;
  el.cam.hidden = false;
  if (speech.Recognition) {
    el.mic.hidden = false;
    speech.stopping = false;
    startRecognition();
    setHint("Listening. Just talk — llmman replies when you stop.");
  } else {
    setHint("Camera on. Type below to talk to the model.");
  }
}

function stopSession() {
  interrupt();
  stopRecognition();
  clearInterval(session.frameTimer);
  session.frameTimer = null;
  if (session.stream) for (const t of session.stream.getTracks()) t.stop();
  session.stream = null;
  session.frames = [];
  el.preview.srcObject = null;
  el.cameraOff.hidden = false;
  el.start.hidden = false;
  el.start.disabled = false;
  el.stop.hidden = true;
  el.mic.hidden = true;
  el.cam.hidden = true;
  setHint("Session ended.");
}

function toggle(button, on, onLabel, offLabel) {
  button.setAttribute("aria-pressed", String(on));
  button.textContent = on ? onLabel : offLabel;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

el.start.addEventListener("click", startSession);
el.stop.addEventListener("click", stopSession);

el.mic.addEventListener("click", () => {
  session.micOn = !session.micOn;
  if (session.micOn) restartRecognition();
  else stopRecognition();
  toggle(el.mic, session.micOn, "Mic on", "Mic off");
});

el.cam.addEventListener("click", () => {
  session.camOn = !session.camOn;
  for (const track of session.stream.getVideoTracks()) {
    track.enabled = session.camOn;
  }
  if (!session.camOn) session.frames = [];
  el.cameraOff.hidden = session.camOn;
  toggle(el.cam, session.camOn, "Camera on", "Camera off");
});

el.compose.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = el.text.value.trim();
  if (!text) return;
  el.text.value = "";
  sendTurn(text);
});

window.addEventListener("pagehide", () => {
  if (session.stream) stopSession();
});

loadModels();
setUpRecognition();
