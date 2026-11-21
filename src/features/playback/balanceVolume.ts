/**
 * App-level loudness leveling for HTML video playback.
 *
 * The Web Audio graph keeps a dry path for instant bypass and a wet path through
 * an AudioWorklet. The worklet measures BS.1770 K-weighted loudness with 400 ms
 * and 3 s windows, applies a slow gain toward the configured LUFS target, and
 * runs a 5 ms look-ahead limiter. A fixed per-track gain can be supplied when a
 * source exposes loudness metadata; that is the NewTube rule: one constant gain
 * per audio track, with the user volume kept separate.
 *
 * Safety boundary: MediaElementAudioSourceNode permanently routes a <video>'s
 * audio into Web Audio and outputs silence if the media element is CORS-opaque
 * (or on iOS native AVPlayer HLS). Only attach when the loaded source is a
 * readable blob:/data:/same-origin/CORS stream.
 */

import Hls from "hls.js";

const WORKLET_URL = new URL("./balanceVolumeProcessor.js", import.meta.url).href;

type MetricsMessage = {
  type: "metrics";
  inputDb?: number;
  gainDb?: number;
  desiredDb?: number;
  reductionDb?: number;
};

type Chain = {
  ctx: AudioContext;
  video: HTMLVideoElement;
  source: MediaElementAudioSourceNode;
  dry: GainNode;
  wet: GainNode;
  outputAnalyser: AnalyserNode;
  outputBuffer: Float32Array<ArrayBuffer>;
  worklet: AudioWorkletNode | null;
  enabled: boolean;
  options: Required<BalanceVolumeOptions>;
  metrics: MetricsMessage | null;
  resetTimer: number;
};

export type BalanceVolumeMetrics = {
  state: "off" | "pending" | "on" | "unsupported";
  inputDb: number | null;
  outputDb: number | null;
  outputPeakDb: number | null;
  levelDb: number | null;
  desiredDb: number;
  gainDb: number;
  reductionDb: number;
};

export type BalanceVolumeOptions = {
  /** Target loudness in LUFS. Default -18; clamped to -24..-14. */
  targetLufs?: number;
  /** Fraction of the error to correct. Default 1; clamped to 0.4..1. */
  strength?: number;
  /** Optional constant per-track gain in dB. Null means use the auto leveler. */
  fixedGainDb?: number | null;
  /** Maximum auto boost in dB. Default 28; clamped to 0..30. */
  maxBoostDb?: number;
  /** Maximum auto cut in dB. Default -18; clamped to -30..0. */
  maxCutDb?: number;
};

const DEFAULT_OPTIONS: Required<BalanceVolumeOptions> = {
  targetLufs: -18,
  strength: 1,
  fixedGainDb: null,
  maxBoostDb: 28,
  maxCutDb: -18,
};

const CROSSFADE_SECONDS = 0.2;
const WORKLET_NAME = "hghub-balance-volume";

const chains = new WeakMap<HTMLVideoElement, Chain>();
const wanted = new WeakMap<HTMLVideoElement, boolean>();
const desiredOptions = new WeakMap<HTMLVideoElement, Required<BalanceVolumeOptions>>();
const pending = new WeakMap<HTMLVideoElement, () => void>();
const pendingSource = new WeakMap<HTMLVideoElement, () => void>();
const attachedRefs: WeakRef<HTMLVideoElement>[] = [];
const workletModules = new WeakMap<AudioContext, Promise<void>>();
let sharedContext: AudioContext | null = null;
let suspendTimer = 0;

function audioContextCtor(): typeof AudioContext | null {
  if (typeof window === "undefined") return null;
  return window.AudioContext
    ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    ?? null;
}

/** True when the runtime supports MSE + Web Audio leveling (hides toggle on iOS native AVPlayer). */
export function isBalanceVolumeSupported(): boolean {
  return Boolean(audioContextCtor())
    && typeof AudioWorkletNode !== "undefined"
    && Hls.isSupported();
}

/** Reject opaque sources before createMediaElementSource permanently silences the element. */
function canSampleElementAudio(video: HTMLVideoElement): "ready" | "empty" | "opaque" {
  const current = video.currentSrc || video.src;
  if (!current) return "empty";
  if (current.startsWith("blob:") || current.startsWith("data:")) return "ready";
  try {
    const url = new URL(current, location.href);
    return url.origin === location.origin || Boolean(video.crossOrigin) ? "ready" : "opaque";
  } catch {
    return "opaque";
  }
}

function anyAttachedVideoPlaying(): boolean {
  for (let i = attachedRefs.length - 1; i >= 0; i -= 1) {
    const el = attachedRefs[i].deref();
    if (!el) {
      attachedRefs.splice(i, 1);
      continue;
    }
    if (el.isConnected && !el.paused && !el.ended) return true;
  }
  return false;
}

function scheduleContextIdleCheck(ctx: AudioContext) {
  if (suspendTimer) window.clearTimeout(suspendTimer);
  suspendTimer = window.setTimeout(() => {
    suspendTimer = 0;
    if (ctx.state === "running" && !anyAttachedVideoPlaying()) {
      void ctx.suspend().catch(() => {});
    }
  }, 320);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function numberOption(value: number | undefined, fallback: number, min: number, max: number) {
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, min, max) : fallback;
}

function normalizeOptions(options?: BalanceVolumeOptions): Required<BalanceVolumeOptions> {
  return {
    targetLufs: numberOption(options?.targetLufs, DEFAULT_OPTIONS.targetLufs, -24, -14),
    strength: numberOption(options?.strength, DEFAULT_OPTIONS.strength, 0.4, 1),
    fixedGainDb: typeof options?.fixedGainDb === "number" && Number.isFinite(options.fixedGainDb)
      ? options.fixedGainDb
      : null,
    maxBoostDb: numberOption(options?.maxBoostDb, DEFAULT_OPTIONS.maxBoostDb, 0, 30),
    maxCutDb: numberOption(options?.maxCutDb, DEFAULT_OPTIONS.maxCutDb, -30, 0),
  };
}

function createContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const Ctor = window.AudioContext
    ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;
  try { return new Ctor({ latencyHint: "interactive" }); }
  catch {
    try { return new Ctor(); }
    catch { return null; }
  }
}

function getContext(): AudioContext | null {
  if (sharedContext?.state === "closed") sharedContext = null;
  if (!sharedContext) sharedContext = createContext();
  return sharedContext;
}

function setState(video: HTMLVideoElement, state: BalanceVolumeMetrics["state"]) {
  video.dataset.balanceVolume = state;
}

function ramp(param: AudioParam, value: number, ctx: AudioContext) {
  const now = ctx.currentTime;
  param.cancelScheduledValues(now);
  param.setValueAtTime(param.value, now);
  param.linearRampToValueAtTime(value, now + CROSSFADE_SECONDS);
}

function levelsFromBuffer(analyser: AnalyserNode, buffer: Float32Array<ArrayBuffer>) {
  analyser.getFloatTimeDomainData(buffer);
  let sum = 0;
  let peak = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    const sample = buffer[i];
    sum += sample * sample;
    const magnitude = Math.abs(sample);
    if (magnitude > peak) peak = magnitude;
  }
  return {
    rmsDb: 20 * Math.log10(Math.sqrt(sum / buffer.length) + 1e-7),
    peakDb: 20 * Math.log10(peak + 1e-7),
  };
}

function loadWorkletModule(ctx: AudioContext) {
  let promise = workletModules.get(ctx);
  if (!promise) {
    // A same-origin asset is allowed by the Tauri CSP (`script-src 'self'`).
    // Blob URLs are intentionally avoided: they would need a `blob:` script-src.
    promise = ctx.audioWorklet.addModule(WORKLET_URL);
    workletModules.set(ctx, promise);
  }
  return promise;
}

function configure(chain: Chain) {
  chain.worklet?.port.postMessage({
    type: "configure",
    target: chain.options.targetLufs,
    strength: chain.options.strength,
    volume: chain.video.volume,
    active: chain.enabled && !chain.video.paused && !chain.video.muted && !chain.video.seeking,
    fixedGainDb: chain.options.fixedGainDb,
    maxBoostDb: chain.options.maxBoostDb,
    maxCutDb: chain.options.maxCutDb,
  });
}

function resetWorklet(chain: Chain) {
  chain.worklet?.port.postMessage({ type: "reset" });
  chain.metrics = null;
}

function enableChain(chain: Chain) {
  chain.enabled = true;
  if (suspendTimer) {
    window.clearTimeout(suspendTimer);
    suspendTimer = 0;
  }
  if (chain.resetTimer) {
    window.clearTimeout(chain.resetTimer);
    chain.resetTimer = 0;
  }
  if (!chain.worklet) {
    // The module is still loading. Keep the dry path audible until it is ready.
    setState(chain.video, "pending");
    void chain.ctx.resume().catch(() => {});
    return;
  }
  resetWorklet(chain);
  configure(chain);
  ramp(chain.dry.gain, 0, chain.ctx);
  ramp(chain.wet.gain, 1, chain.ctx);
  setState(chain.video, "on");
  void chain.ctx.resume().catch(() => {});
}

function disableChain(chain: Chain) {
  chain.enabled = false;
  ramp(chain.dry.gain, 1, chain.ctx);
  ramp(chain.wet.gain, 0, chain.ctx);
  setState(chain.video, "off");
  if (chain.resetTimer) window.clearTimeout(chain.resetTimer);
  chain.resetTimer = window.setTimeout(() => {
    chain.resetTimer = 0;
    resetWorklet(chain);
    configure(chain);
    scheduleContextIdleCheck(chain.ctx);
  }, 300);
}

function clearPending(video: HTMLVideoElement) {
  const cleanup = pending.get(video);
  if (cleanup) {
    cleanup();
    pending.delete(video);
  }
  const sourceCleanup = pendingSource.get(video);
  if (sourceCleanup) {
    sourceCleanup();
    pendingSource.delete(video);
  }
}

function waitForReadableSource(video: HTMLVideoElement, ctx: AudioContext, options: Required<BalanceVolumeOptions>) {
  if (pendingSource.has(video)) return;
  const onSource = () => {
    cleanup();
    if (wanted.get(video)) void attach(video, ctx, desiredOptions.get(video) ?? options);
  };
  const cleanup = () => {
    video.removeEventListener("loadedmetadata", onSource);
    video.removeEventListener("play", onSource);
    pendingSource.delete(video);
  };
  video.addEventListener("loadedmetadata", onSource, { once: true });
  video.addEventListener("play", onSource, { once: true });
  pendingSource.set(video, cleanup);
}

function waitForGesture(video: HTMLVideoElement, ctx: AudioContext, options: Required<BalanceVolumeOptions>) {
  if (pending.has(video)) return;
  let done = false;
  const events = ["pointerdown", "keydown", "touchend"] as const;
  const cleanup = () => {
    for (const event of events) document.removeEventListener(event, unlock, true);
  };
  const unlock = () => {
    if (done) return;
    done = true;
    cleanup();
    pending.delete(video);
    void ctx.resume().then(() => {
      if (wanted.get(video)) void attach(video, ctx, desiredOptions.get(video) ?? options);
    }).catch(() => { /* Leave the dry path active; the setting stays pending. */ });
  };
  for (const event of events) document.addEventListener(event, unlock, true);
  pending.set(video, cleanup);
}

function readMetrics(value: unknown): MetricsMessage | null {
  if (!value || typeof value !== "object") return null;
  const data = value as MetricsMessage;
  if (data.type !== "metrics") return null;
  return data;
}

async function attach(video: HTMLVideoElement, ctx: AudioContext, options: Required<BalanceVolumeOptions>) {
  if (!wanted.get(video)) return;
  const existing = chains.get(video);
  if (existing) {
    existing.options = options;
    enableChain(existing);
    return;
  }

  const eligibility = canSampleElementAudio(video);
  if (eligibility === "empty") {
    waitForReadableSource(video, ctx, options);
    return;
  }
  if (eligibility === "opaque") {
    clearPending(video);
    setState(video, "unsupported");
    return;
  }
  clearPending(video);

  let source: MediaElementAudioSourceNode;
  try {
    source = ctx.createMediaElementSource(video);
  } catch {
    setState(video, "unsupported");
    return;
  }

  const dry = ctx.createGain();
  dry.gain.value = 1;
  const wet = ctx.createGain();
  wet.gain.value = 0;
  const outputAnalyser = ctx.createAnalyser();
  outputAnalyser.fftSize = 2048;
  outputAnalyser.smoothingTimeConstant = 0;

  // Keep the dry path audible while the worklet module loads.
  source.connect(dry);
  dry.connect(ctx.destination);

  const chain: Chain = {
    ctx, video, source, dry, wet, outputAnalyser,
    outputBuffer: new Float32Array(outputAnalyser.fftSize),
    worklet: null, enabled: false, options, metrics: null, resetTimer: 0,
  };
  chains.set(video, chain);
  attachedRefs.push(new WeakRef(video));

  const resync = () => {
    resetWorklet(chain);
    configure(chain);
  };
  video.addEventListener("play", () => {
    if (suspendTimer) {
      window.clearTimeout(suspendTimer);
      suspendTimer = 0;
    }
    void ctx.resume().catch(() => {});
    configure(chain);
  });
  const onStop = () => {
    configure(chain);
    scheduleContextIdleCheck(ctx);
  };
  video.addEventListener("pause", onStop);
  video.addEventListener("ended", onStop);
  video.addEventListener("volumechange", () => configure(chain));
  video.addEventListener("seeking", resync);
  video.addEventListener("seeked", resync);
  video.addEventListener("emptied", resync);
  video.addEventListener("loadedmetadata", resync);
  video.addEventListener("loadstart", resync);

  try {
    await loadWorkletModule(ctx);
  } catch {
    setState(video, "unsupported");
    return;
  }
  if (!wanted.get(video)) return;

  try {
    const worklet = new AudioWorkletNode(ctx, WORKLET_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      channelCount: 2,
      channelCountMode: "explicit",
    });
    worklet.port.onmessage = (event) => {
      const metrics = readMetrics(event.data);
      if (metrics) chain.metrics = metrics;
    };
    worklet.onprocessorerror = () => {
      chain.worklet = null;
      setState(video, "unsupported");
      ramp(chain.dry.gain, 1, ctx);
      ramp(chain.wet.gain, 0, ctx);
      try { worklet.disconnect(); } catch { /* already detached */ }
      scheduleContextIdleCheck(ctx);
    };

    source.connect(worklet);
    worklet.connect(outputAnalyser);
    outputAnalyser.connect(wet);
    wet.connect(ctx.destination);
    chain.worklet = worklet;
    enableChain(chain);
  } catch {
    setState(video, "unsupported");
    scheduleContextIdleCheck(ctx);
  }
}

/** Apply the persisted preference to a video element. Creating the source node is lazy. */
export function setBalanceVolume(
  video: HTMLVideoElement | null,
  enabled: boolean,
  options?: BalanceVolumeOptions,
) {
  if (!video) return;
  const normalized = normalizeOptions(options);
  wanted.set(video, enabled);
  desiredOptions.set(video, normalized);

  const existing = chains.get(video);
  if (!enabled) {
    clearPending(video);
    if (existing) disableChain(existing);
    else setState(video, "off");
    return;
  }

  const eligibility = canSampleElementAudio(video);
  if (eligibility === "opaque") {
    clearPending(video);
    if (existing) disableChain(existing);
    setState(video, "unsupported");
    return;
  }

  if (existing) {
    existing.options = normalized;
    enableChain(existing);
    return;
  }

  const ctx = getContext();
  if (!ctx) { setState(video, "unsupported"); return; }
  setState(video, "pending");
  const activate = () => {
    if (wanted.get(video) && ctx.state === "running") void attach(video, ctx, normalized);
  };
  void ctx.resume().then(activate).catch(() => { /* Wait for the next explicit gesture. */ });
  if (ctx.state !== "running") waitForGesture(video, ctx, normalized);
  else activate();
}

/** Test-only observability for the live audio graph. */
export function getBalanceVolumeMetrics(video: HTMLVideoElement | null): BalanceVolumeMetrics | null {
  if (!video) return null;
  const chain = chains.get(video);
  if (!chain) {
    return {
      state: (video.dataset.balanceVolume as BalanceVolumeMetrics["state"]) ?? "off",
      inputDb: null, outputDb: null, outputPeakDb: null, levelDb: null, desiredDb: 0, gainDb: 0,
      reductionDb: 0,
    };
  }
  const output = levelsFromBuffer(chain.outputAnalyser, chain.outputBuffer);
  const metrics = chain.metrics;
  return {
    state: (video.dataset.balanceVolume as BalanceVolumeMetrics["state"]) ?? "off",
    inputDb: metrics?.inputDb ?? null,
    outputDb: output.rmsDb,
    outputPeakDb: output.peakDb,
    levelDb: metrics?.inputDb ?? null,
    desiredDb: metrics?.desiredDb ?? 0,
    gainDb: metrics?.gainDb ?? 0,
    reductionDb: metrics?.reductionDb ?? 0,
  };
}
