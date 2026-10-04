// Voice: speech to text, and replies read aloud.
//
// Listening uses the browser's recogniser where there is one (Chrome and
// Edge send the audio to their vendor). Elsewhere (Firefox, Android's
// WebView) it records the microphone and has the daemon transcribe it,
// which needs a model that takes audio input. Speaking uses the browser's
// speech synthesis; whether a voice is local is up to the browser.

import { deviceError } from "./attach.js";

const Recognition = globalThis.SpeechRecognition || globalThis.webkitSpeechRecognition;

/** Whether the page can listen: a recogniser, or a microphone and recorder. */
export function canListen() {
  const canRecord =
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined" &&
    typeof AudioContext !== "undefined" &&
    typeof OfflineAudioContext !== "undefined"; // toWav needs both
  return !!Recognition || canRecord;
}

export function canSpeak() {
  return typeof speechSynthesis !== "undefined" && typeof SpeechSynthesisUtterance !== "undefined";
}

// ---- Listening ----------------------------------------------------------------

const RECOGNITION_ERRORS = {
  "not-allowed": "Allow microphone access to use voice input",
  "service-not-allowed": "Allow microphone access to use voice input",
  "audio-capture": "No microphone found",
  network: "This browser's voice input needs a network connection",
  "language-not-supported": "This browser's voice input does not support your language",
};

/** Callbacks that run once, and not after `cancel()`, which also aborts `signal`. */
function session() {
  let over = false;
  const abort = new AbortController();
  return {
    signal: abort.signal,
    get over() {
      return over;
    },
    end(fn) {
      if (over) return;
      over = true;
      fn();
    },
    cancel() {
      over = true;
      abort.abort();
    },
  };
}

/**
 * Listens for one utterance. `onState("listening" | "transcribing")`,
 * `onInterim(text)` (the browser's recogniser only), then once `onDone(text)`
 * ("" for none) or `onError(message)`. `transcribe(wav, signal)` resolves
 * with the text of a recording, and is only used when there is no
 * recogniser; `signal` aborts if the listening is cancelled.
 * Returns `{stop, cancel}`: `stop` ends the utterance and still reports it.
 */
export function listen({ onState, onInterim, onDone, onError, transcribe }) {
  if (Recognition) return listenNative({ onState, onInterim, onDone, onError });
  if (canListen() && transcribe) return listenRecorded({ onState, onDone, onError, transcribe });
  queueMicrotask(() => onError("Voice input is not available in this browser"));
  return { stop() {}, cancel() {} };
}

function listenNative({ onState, onInterim, onDone, onError }) {
  const s = session();
  const recognition = new Recognition();
  recognition.lang = navigator.language || "en-US";
  recognition.interimResults = true;
  recognition.continuous = false; // one utterance, ended by the speaker's pause
  let heard = "";
  recognition.onresult = (e) => {
    heard = Array.from(e.results, (r) => r[0].transcript).join("").trim();
    onInterim?.(heard);
  };
  recognition.onerror = (e) => {
    if (e.error === "no-speech" || e.error === "aborted") return; // onend reports these
    s.end(() => onError(RECOGNITION_ERRORS[e.error] || `Voice input failed: ${e.error}`));
  };
  recognition.onend = () => s.end(() => onDone(heard));
  try {
    recognition.start();
    onState?.("listening");
  } catch (e) {
    queueMicrotask(() => s.end(() => onError(`Voice input failed: ${e.message || e}`)));
  }
  return {
    stop: () => recognition.stop(),
    cancel: () => {
      s.cancel();
      recognition.abort();
    },
  };
}

const VOICE_RMS = 0.02; // microphone loudness (0..1) taken to be speech
const PAUSE_MS = 1500; // quiet after speech that ends the utterance
const LEAD_MS = 8000; // quiet at the start that gives up
const MAX_MS = 60_000;

function listenRecorded({ onState, onDone, onError, transcribe }) {
  const s = session();
  let stopRecording = () => {};
  let stopWanted = false; // stop() came before the recorder was running

  (async () => {
    let stream, audio, timer;
    const release = () => {
      clearInterval(timer);
      stream?.getTracks().forEach((track) => track.stop());
      audio?.close().catch(() => {});
    };
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true }).catch((e) => {
        throw new Error(deviceError(e, "microphone"));
      });
      if (s.over) return;
      onState?.("listening");

      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      const stopped = new Promise((resolve) => (recorder.onstop = resolve));
      let failure = null; // a recorder error is followed by `stop`
      recorder.onerror = (e) => (failure = e.error || new Error("The recording failed"));

      // The recording ends at a pause after speech, or at the silence of
      // an utterance that never began.
      audio = new AudioContext();
      const analyser = audio.createAnalyser();
      analyser.fftSize = 1024;
      audio.createMediaStreamSource(stream).connect(analyser);
      const samples = new Uint8Array(analyser.fftSize);
      const started = performance.now();
      let lastVoice = started;
      let heardVoice = false;
      let silent = false;
      stopRecording = () => recorder.state !== "inactive" && recorder.stop();
      timer = setInterval(() => {
        analyser.getByteTimeDomainData(samples);
        let sum = 0;
        for (const v of samples) sum += ((v - 128) / 128) ** 2;
        const now = performance.now();
        if (Math.sqrt(sum / samples.length) > VOICE_RMS) {
          heardVoice = true;
          lastVoice = now;
        }
        silent = !heardVoice && now - started > LEAD_MS;
        if ((heardVoice && now - lastVoice > PAUSE_MS) || silent || now - started > MAX_MS) stopRecording();
      }, 100);

      recorder.start();
      if (stopWanted) stopRecording();
      await stopped;
      release(); // the microphone is off before the transcription starts
      if (failure) throw failure;
      if (s.over) return;
      if (silent || !chunks.length) return s.end(() => onDone(""));

      onState?.("transcribing");
      const text = await transcribe(await toWav(new Blob(chunks, { type: recorder.mimeType })), s.signal);
      s.end(() => onDone(text));
    } catch (e) {
      s.end(() => onError(e.message || String(e)));
    } finally {
      release();
    }
  })();

  return {
    stop: () => {
      stopWanted = true;
      stopRecording();
    },
    cancel: () => {
      s.cancel();
      stopRecording();
    },
  };
}

// ---- Audio ---------------------------------------------------------------------

const SPEECH_RATE = 16_000; // what speech models are trained on

/** A browser recording (webm/opus, mp4) as a 16 kHz mono wav, which llama.cpp reads. */
async function toWav(recording) {
  const context = new AudioContext();
  let decoded;
  try {
    decoded = await context.decodeAudioData(await recording.arrayBuffer());
  } catch {
    throw new Error("This browser could not read its own recording");
  } finally {
    context.close().catch(() => {});
  }
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * SPEECH_RATE)), SPEECH_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination); // mixes down to mono
  source.start();
  return encodeWav((await offline.startRendering()).getChannelData(0), SPEECH_RATE);
}

/** Mono float samples in -1..1 as a 16-bit PCM wav Blob. */
export function encodeWav(samples, rate) {
  const bytes = samples.length * 2;
  const view = new DataView(new ArrayBuffer(44 + bytes));
  const text = (offset, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + bytes, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true); // bytes per second
  view.setUint16(32, 2, true); // bytes per frame
  view.setUint16(34, 16, true); // bits per sample
  text(36, "data");
  view.setUint32(40, bytes, true);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  return new Blob([view.buffer], { type: "audio/wav" });
}

// ---- Speaking --------------------------------------------------------------------

/** Markdown as it should sound: no code, link addresses, emphasis or [1] citations. */
export function speakable(markdown) {
  return markdown
    .replace(/```[\s\S]*?(```|$)/g, " ") // a code block, even an unfinished one
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[\d+\]/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/^\s*([-*+]|\d+[.)])\s+/gm, "")
    .replace(/^\s*\|?(\s*:?-{3,}:?\s*\|?)+\s*$/gm, " ") // a table's rule row
    .replace(/\|/g, ", ")
    .replace(/\*\*|__|~~|\*/g, "")
    .replace(/[ \t]+/g, " ")
    .replace(/\n\s*\n+/g, "\n")
    .trim();
}

/**
 * `text` as utterances of at most `max` characters, split at sentence
 * ends: Chrome cuts an utterance off after about fifteen seconds.
 */
export function speechChunks(text, max = 160) {
  const pieces = [];
  for (const line of speakable(text).split("\n")) {
    for (const sentence of line.match(/.+?(?:[。！？]+|[.!?]+(?=\s|$)|$)/g) || []) {
      let rest = sentence.trim();
      while (rest.length > max) {
        const cut = rest.lastIndexOf(" ", max);
        pieces.push(rest.slice(0, cut > 0 ? cut : max).trim());
        rest = rest.slice(cut > 0 ? cut : max).trim();
      }
      if (rest) pieces.push(rest);
    }
  }
  const chunks = [];
  let current = "";
  for (const piece of pieces) {
    if (current && current.length + 1 + piece.length > max) {
      chunks.push(current);
      current = piece;
    } else {
      current = current ? `${current} ${piece}` : piece;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

let speech = null; // the speech in progress: its timer and utterances
const STALL_MS = 4000; // for a chunk to start
const PLAYBACK_MS = 30_000; // for one chunk (160 characters) to be said

/**
 * Reads `text` (markdown) aloud. `onEnd(error?)` runs once when it has
 * been said or could not be, not when `stopSpeaking` or a later `speak`
 * cut it off.
 */
export function speak(text, onEnd) {
  stopSpeaking();
  const chunks = speechChunks(text);
  if (!chunks.length || !canSpeak()) {
    queueMicrotask(() => onEnd?.(canSpeak() ? undefined : "Speech output is not available in this browser"));
    return;
  }
  const mine = {};
  speech = mine;
  const done = (error) => {
    if (speech !== mine) return;
    speech = null;
    clearTimeout(mine.timer);
    if (error) speechSynthesis.cancel(); // the rest of the queue must not carry on
    onEnd?.(error);
  };
  // An engine can accept speak() and then never start, stall between
  // chunks or never finish one, without reporting anything: a hands-free
  // conversation would hang.
  const arm = (ms = STALL_MS) => {
    clearTimeout(mine.timer);
    mine.timer = setTimeout(() => done("Speech output is not responding"), ms);
  };
  arm();
  // Kept on `mine`: Chrome drops an utterance nothing references, and its events with it.
  mine.utterances = chunks.map((chunk, i) => {
    const utterance = new SpeechSynthesisUtterance(chunk);
    utterance.onstart = () => arm(PLAYBACK_MS);
    utterance.onend = () => (i === chunks.length - 1 ? done() : arm());
    utterance.onerror = (e) => {
      if (e.error !== "canceled" && e.error !== "interrupted") done(`Speech output failed: ${e.error}`);
    };
    return utterance;
  });
  for (const utterance of mine.utterances) speechSynthesis.speak(utterance);
}

/** Stops what `speak` is saying; its `onEnd` is not called. */
export function stopSpeaking() {
  if (!speech) return;
  clearTimeout(speech.timer);
  speech = null;
  speechSynthesis.cancel();
}
