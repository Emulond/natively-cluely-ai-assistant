/**
 * LocalWhisperSTT — local Whisper / Distil-Whisper / Moonshine STT provider.
 *
 * Dual-channel architecture: Natively captures Mic and System Audio as two
 * completely separate native streams. createSTTProvider() instantiates this
 * class TWICE — once per channel. No diarization model is needed; speaker
 * attribution is free from the hardware.
 *
 * STREAMING DESIGN (closes the latency gap with cloud STT):
 *
 *   Cloud STT providers (Deepgram/Soniox/ElevenLabs) emit *interim*
 *   transcripts every 100–300ms while the user is still speaking. Whisper
 *   wasn't designed for streaming — we approximate it with a per-model
 *   profile (see resolveStreamingProfile):
 *
 *   Whisper / Distil-Whisper path (slow, batch-architected models):
 *     - Tick every 1500ms while a segment is open (after 800ms of audio)
 *     - Apply LocalAgreement-2: only commit text where two overlapping
 *       inferences agree (longest common prefix). Stabilizes flicker.
 *     - First interim emit ~1.5–2.5s after speech starts.
 *
 *   Moonshine path (streaming-native, deterministic, ~100ms inference):
 *     - Tick every 750ms after just 400ms of audio
 *     - Skip LA-2 — the model's output is already stable; emit each
 *       cleaned partial directly.
 *     - First interim emit ~400–600ms after speech starts.
 *
 *   When VAD closes the segment (or hits MAX_SEGMENT_MS for a soft commit):
 *     - Run a final pass on the full segment
 *     - Emit { isFinal: true, confidence: 0.9 }
 *     - Reset session state for the next segment
 */

import { EventEmitter } from 'events';
import { Worker } from 'worker_threads';
import { resampleToF32 } from './whisper/audioResampler';
import { VadProcessor } from './whisper/vadProcessor';
import { filterHallucination } from './whisper/hallucinationFilter';
import {
    configureTransformersCache,
    getCpuRealtimeModelNames,
    getModelDisplayName,
    getModelSizeBytes,
    getMultilingualModelNames,
    isMultilingualModel,
    isTooSlowForCpuRealtime,
} from './whisper/modelManager';
import { clearLoadSentinel, modelPreloader, writeLoadSentinel } from './whisper/modelPreloader';
import { buildWorkerInitMessage } from './whisper/inferenceConfig';
import { resolveWhisperWorkerPath } from './whisper/workerPathResolver';
import type { WorkerOutMessage } from './whisper/types';
import { acquireOnnxSlot, hasEnoughMemoryForOnnxSession, getMinFreeGBForOnnxSession, getAvailableMemoryGB } from '../utils/onnxThreadConfig';

export class LocalWhisperSTT extends EventEmitter {
    private readonly modelId: string;
    private inputSampleRate = 48000;
    private language = 'auto';
    // Optional context-biasing prompt sent out-of-band to the worker via
    // `setPrompt` messages. The worker tokenizes once and reuses the IDs
    // for every transcribe (see whisperWorker.ts updatePromptCache). 224
    // Whisper-decoder tokens cap is enforced worker-side. No-op for Moonshine.
    private contextPrompt = '';
    private contextPromptSentToWorker = '';
    // Char-length cap to prevent enormous strings from being copied through
    // worker IPC. ~8KB is well above 224 Whisper tokens (~3-4 chars/token).
    private static readonly PROMPT_MAX_CHARS = 8000;

    // ── Latency telemetry ──────────────────────────────────────────────
    // Perceived latency tracking. Two metrics:
    //   firstPartial = ms from VAD opening a segment → first agreed/committed
    //                  prefix emit (LocalAgreement-2 needs two streaming ticks
    //                  to converge, so this is NOT "first inference time").
    //   final        = ms from VAD opening a segment → final transcript emit.
    // Boundary detection uses VadProcessor.currentSegmentId() (monotonic
    // counter) instead of boolean edges on isInSpeech() — boolean edges miss
    // open+close-in-one-push and close+open-in-one-push patterns.
    private trackedSegmentId = 0;
    private segmentOpenedAt = 0;
    private firstPartialEmittedForSegment = 0;
    private firstPartialLatencies: number[] = [];
    private finalLatencies: number[] = [];
    private static readonly LATENCY_WINDOW = 100;
    private static readonly LATENCY_LOG_EVERY = 20;
    // Sanity clamp: any latency outside this range is treated as a tracking
    // bug (e.g. clock issue, missed segment id) and discarded so it can't
    // pollute p95/p99.
    private static readonly LATENCY_MAX_MS = 60_000;
    private latencyLogCounter = 0;
    // Optional channel label ('mic' / 'system') — disambiguates log lines
    // when both LocalWhisperSTT instances run the same model.
    private channelLabel = '';

    // ── Pipeline diagnostics ────────────────────────────────────────────────
    // The audio path has three points that drop a chunk with no trace:
    //   1. write()        — returns early when !isActive || !vad
    //   2. VadProcessor   — emits no segment when RMS stays under 0.008
    //   3. dispatchFinal  — returns early when !worker
    // All three produce the same user-visible symptom (audio captured, no
    // transcript, no error), so a report that cannot distinguish them is
    // useless. These counters attribute the loss to exactly one stage, and
    // carry the observed signal level so "too quiet for VAD" is separable
    // from "VAD fine, nothing downstream".
    private diagChunks = 0;
    private diagDroppedInactive = 0;
    private diagDroppedNoWorker = 0;
    private diagDroppedBackpressure = 0;
    private diagSegments = 0;
    private diagDispatched = 0;
    private diagPeakRms = 0;
    private diagLastReportAt = 0;
    private static readonly DIAG_REPORT_MS = 5000;

    /** Cheap strided RMS over the resampled window, for VAD-threshold comparison. */
    private static rmsOf(samples: Float32Array): number {
        if (samples.length === 0) return 0;
        const stride = Math.max(1, samples.length >> 8);
        let sum = 0, n = 0;
        for (let i = 0; i < samples.length; i += stride) { sum += samples[i] * samples[i]; n++; }
        return n ? Math.sqrt(sum / n) : 0;
    }

    private diagReport(force = false): void {
        const now = performance.now();
        if (this.diagLastReportAt === 0) this.diagLastReportAt = now;
        if (!force && now - this.diagLastReportAt < LocalWhisperSTT.DIAG_REPORT_MS) return;
        this.diagLastReportAt = now;
        const tag = this.channelLabel ? `:${this.channelLabel}` : '';
        // peakRms vs VAD's 0.008 threshold is the single most diagnostic number
        // here: above it with segments=0 means VAD is misbehaving; below it
        // means the mic signal is genuinely too quiet to be treated as speech.
        console.log(
            `[LocalWhisperSTT${tag}] pipeline · chunks=${this.diagChunks}` +
            ` peakRms=${this.diagPeakRms.toFixed(5)} (vadThreshold=0.008)` +
            ` segments=${this.diagSegments} dispatched=${this.diagDispatched}` +
            ` pending=${this.pendingAudio.length}` +
            ` droppedInactive=${this.diagDroppedInactive} droppedNoWorker=${this.diagDroppedNoWorker}` +
            ` droppedBackpressure=${this.diagDroppedBackpressure} inFlight=${this.inFlightTranscribes}` +
            // The worker cannot report its own elapsed time while ONNX is
            // running (its timers starve), so name the outstanding tasks here.
            // "inFlight=2 stuck=t1,t2" is the signature of a model too slow for
            // this machine; "inFlight=2" with the ids changing every line is
            // just a busy pipeline.
            (this.taskWatchdogs.size > 0
                ? ` outstanding=${[...this.taskWatchdogs.keys()].join(',')}`
                : '') +
            ` | active=${this.isActive} vad=${!!this.vad} worker=${!!this.worker} ready=${this.workerReady}`,
        );
        this.diagChunks = 0;
        this.diagDroppedInactive = 0;
        this.diagDroppedNoWorker = 0;
        this.diagDroppedBackpressure = 0;
        this.diagSegments = 0;
        this.diagDispatched = 0;
        this.diagPeakRms = 0;
    }
    private worker: Worker | null = null;
    private vad: VadProcessor | null = null;
    private isActive = false;
    // Cross-loader ONNX gate slot. Acquired in spawnWorker() before posting
    // init; released in worker error/exit handlers so other ONNX consumers
    // (LocalReranker / LocalEmbeddingProvider / IntentClassifier) can take
    // the slot promptly. Whisper uses priority 'high' so its streaming loop
    // acquires before queued normal-priority consumers.
    private slotRelease: (() => void) | null = null;

    // Count of Whisper workers alive in THIS process. worker_threads share the
    // process heap, so this is the number of full model copies resident —
    // the quantity the second-session admission check above reasons about.
    private static liveWorkers = 0;
    private countedLive = false;

    private markWorkerLive(): void {
        if (this.countedLive) return;
        this.countedLive = true;
        LocalWhisperSTT.liveWorkers++;
    }

    private markWorkerGone(): void {
        if (!this.countedLive) return;
        this.countedLive = false;
        LocalWhisperSTT.liveWorkers = Math.max(0, LocalWhisperSTT.liveWorkers - 1);
    }
    private taskCounter = 0;
    private workerReady = false;
    private isDrainingFinals = false;
    private drainingFinalsInFlight = 0;
    // Pending audio waiting for the worker to become ready. Always finals —
    // streaming partials are never queued (they're best-effort and only fire
    // while a segment is open AND the worker is ready).
    private pendingAudio: Float32Array[] = [];

    // Gap-flush: ensures a segment closes even if Rust SilenceSuppressor
    // stops sending audio before VAD's hangover completes.
    private gapFlushTimer: ReturnType<typeof setTimeout> | null = null;
    private static readonly GAP_FLUSH_MS = 400;
    // 5s grace timer for the previous worker to finish in-flight transcribes
    // before we terminate it. Tracked so rapid stop/start cycles or app quit
    // don't pin the event loop with stale termination timers.
    private workerTerminateTimer: ReturnType<typeof setTimeout> | null = null;

    // Streaming inference loop state.
    // Self-chaining setTimeout (not setInterval) so the delay can adapt at
    // each tick — the worker can be slower than STREAMING_INTERVAL_MS for
    // larger models (whisper-medium ~3-5s, whisper-large ~5-10s); piling up
    // ticks against an in-flight inference just churns the JS event loop.
    private streamingTimer: ReturnType<typeof setTimeout> | null = null;
    // Tuned per model family at construction time (see resolveStreamingProfile).
    private readonly streamingIntervalBaseMs: number;
    private readonly streamingMinAudioMs: number;
    private readonly skipAgreement: boolean;
    private static readonly STREAMING_INTERVAL_MAX_MS = 12000;
    private static readonly MAX_SEGMENT_MS = 14000;       // soft-commit before VAD's 15s hard-flush
    // Backoff: count consecutive ticks where we couldn't dispatch (worker
    // busy or no open segment with enough audio). After 3 in a row, double
    // the next delay; reset to base on a successful dispatch.
    private streamingStallCount = 0;
    private streamingNextDelayMs = 0; // set in constructor from streamingIntervalBaseMs
    // Watchdog: if the worker takes longer than this on an in-flight streaming
    // task we assume it's stuck (hypothesis: GPU lock, deadlock, dead pointer)
    // and force-clear the in-flight state so the loop can recover. Without
    // this, a stuck worker permanently pins streamingTaskInFlight=true and
    // every subsequent tick is a no-op stall (transcription appears to stop
    // after 3-4 questions once the worker gets wedged).
    private streamingWatchdogTimer: ReturnType<typeof setTimeout> | null = null;
    private static readonly STREAMING_WATCHDOG_MS = 30000;

    // LocalAgreement-2 state. We hold the last partial transcript, and when
    // the next partial arrives we emit the longest common prefix as the
    // "stable" interim. The lastEmittedText is what we've already shown.
    private lastPartialText = '';
    private lastEmittedText = '';
    private streamingTaskInFlight = false;
    private streamingTaskId: string | null = null;

    constructor(modelId: string) {
        super();
        this.modelId = modelId;
        configureTransformersCache();

        // Tune the streaming loop for this specific model's characteristics.
        // Moonshine: ~100ms inference, deterministic single-pass output, no
        // 30s padding. We can poll faster, dispatch on shorter audio, and
        // skip LocalAgreement-2's two-pass stability check (which adds an
        // entire tick of latency).
        // Whisper / Distil-Whisper: ~500ms-5s inference, conservative
        // params, LA-2 needed for stability.
        const profile = LocalWhisperSTT.resolveStreamingProfile(modelId);
        this.streamingIntervalBaseMs = profile.intervalMs;
        this.streamingMinAudioMs = profile.minAudioMs;
        this.skipAgreement = profile.skipAgreement;
        this.streamingNextDelayMs = this.streamingIntervalBaseMs;
        console.log(`[LocalWhisperSTT] streaming profile for ${modelId}: interval=${profile.intervalMs}ms minAudio=${profile.minAudioMs}ms skipAgreement=${profile.skipAgreement}`);
    }

    /**
     * Per-model streaming-loop profile. Faster, more aggressive parameters
     * for streaming-class models (Moonshine) — they finish each pass in
     * <200ms and produce stable output, so we can poll often and emit
     * partials directly without LocalAgreement-2's two-pass confirmation.
     */
    private static resolveStreamingProfile(modelId: string): { intervalMs: number; minAudioMs: number; skipAgreement: boolean } {
        // Loose match — covers `onnx-community/moonshine-*`, `usefulsensors/
        // moonshine-*`, and any future fork that keeps "moonshine" in the
        // path. Falls back to Whisper-safe defaults on no match.
        // TODO: validate the 750/400 numbers against measured first-partial
        // p50 once a Moonshine model is downloaded; expect <600ms.
        if (modelId.toLowerCase().includes('moonshine')) {
            return { intervalMs: 750, minAudioMs: 400, skipAgreement: true };
        }
        return { intervalMs: 1500, minAudioMs: 800, skipAgreement: false };
    }

    setSampleRate(rate: number): void { this.inputSampleRate = rate; }
    setAudioChannelCount(_count: number): void {}
    setRecognitionLanguage(key: string): void {
        this.language = key || 'auto';
        this.warnIfLanguageUnsupportedByModel();
    }

    /**
     * Say so, in the log, when the chosen language and the chosen model cannot
     * work together. Selecting Russian on an English-only checkpoint produces
     * either English words or no words at all, and both look like a broken
     * transcriber rather than an impossible pairing.
     */
    private warnIfLanguageUnsupportedByModel(): void {
        const lang = this.language;
        if (!lang || lang === 'auto' || lang.startsWith('english')) return;
        if (isMultilingualModel(this.modelId)) return;
        const tag = this.channelLabel ? `:${this.channelLabel}` : '';
        console.warn(
            `[LocalWhisperSTT${tag}] language "${lang}" was selected but ${this.modelId} is ` +
            `English-only — it cannot produce ${lang} no matter how clear the audio is. ` +
            `Choose a multilingual model in Settings → Audio (${getMultilingualModelNames().join(', ')}).`,
        );
    }
    setCredentials(_credPath: string): void {}

    /**
     * Optional human-readable channel label (e.g. 'mic', 'system') for log
     * disambiguation when both LocalWhisperSTT instances use the same model.
     */
    setChannel(label: string): void { this.channelLabel = (label ?? '').trim(); }

    /**
     * Set a context-biasing prompt (proper nouns, jargon, attendee names).
     * Pushed to the worker out-of-band only when the value actually changes.
     * Empty string disables biasing. Worker truncates to 224 Whisper tokens
     * (front of string preserved) and skips entirely for Moonshine. Safe to
     * call mid-stream — the worker applies the new prompt to subsequent
     * transcribes only; the in-flight one continues with the previous cache.
     */
    setContext(prompt: string): void {
        let trimmed = (prompt ?? '').trim();
        if (trimmed.length > LocalWhisperSTT.PROMPT_MAX_CHARS) {
            trimmed = trimmed.slice(0, LocalWhisperSTT.PROMPT_MAX_CHARS);
        }
        this.contextPrompt = trimmed;
        this.maybePushPromptToWorker();
    }

    private maybePushPromptToWorker(): void {
        if (!this.worker || !this.workerReady) return; // pushed in flushPending after ready
        if (this.contextPrompt === this.contextPromptSentToWorker) return;
        this.worker.postMessage({ type: 'setPrompt', prompt: this.contextPrompt });
        this.contextPromptSentToWorker = this.contextPrompt;
    }

    start(): void {
        if (this.isActive) return;
        this.isDrainingFinals = false;
        this.drainingFinalsInFlight = 0;
        this.isActive = true;
        // Repeat here as well as in setRecognitionLanguage: by now the channel
        // label is set, so the warning says which channel is misconfigured.
        this.warnIfLanguageUnsupportedByModel();
        this.warnIfModelTooSlowForThisMachine();
        this.vad = new VadProcessor();
        this.spawnWorker().catch((err) => {
            // Gate refusal or worker spawn failure (e.g. insufficient memory for
            // the ONNX session). There is NO retry path, so we must NOT leave a
            // live streaming loop + VAD churning with worker=null — that silently
            // drops every audio segment (dispatchFinal early-returns on !worker)
            // and leaks a self-chaining 12s streaming timer for the whole session.
            // Tear the instance back down to a clean inactive no-op (write() then
            // no-ops on !isActive/!vad) and surface the error so the supervisor
            // can fall back to cloud STT.
            console.error('[LocalWhisperSTT] spawnWorker failed:', err);
            this.stopStreamingLoop();
            if (this.gapFlushTimer) {
                clearTimeout(this.gapFlushTimer);
                this.gapFlushTimer = null;
            }
            this.vad = null;
            this.isActive = false;
            this.workerReady = false;
            this.emit('error', err instanceof Error ? err : new Error(String(err)));
        });
        this.startStreamingLoop();
    }

    stop(): void {
        if (!this.isActive) return;
        this.isActive = false;

        this.stopStreamingLoop();
        if (this.gapFlushTimer) {
            clearTimeout(this.gapFlushTimer);
            this.gapFlushTimer = null;
        }

        if (this.vad) {
            const segs = this.vad.flush();
            this.vad = null;
            this.isDrainingFinals = true;
            segs.forEach(s => this.dispatchFinal(s.samples));
            // Held audio is real speech the user said — send it with the rest
            // of the drain rather than discarding it on the way out.
            this.flushCoalesced();
        }

        this.resetAgreementState();

        // Print one final latency summary for the just-ended session, then
        // reset windows so the next start() starts with a clean slate.
        if (this.firstPartialLatencies.length > 0 || this.finalLatencies.length > 0) {
            this.logLatencySummary();
        }
        this.firstPartialLatencies = [];
        this.finalLatencies = [];
        this.segmentOpenedAt = 0;
        this.firstPartialEmittedForSegment = 0;
        this.trackedSegmentId = 0;
        this.latencyLogCounter = 0;

        const w = this.worker;
        if (w) {
            const shouldKeepWorkerForFinals = this.isDrainingFinals && (this.pendingAudio.length > 0 || this.drainingFinalsInFlight > 0);
            if (shouldKeepWorkerForFinals) return;
            this.beginWorkerTermination(w);
        }
    }

    write(chunk: Buffer): void {
        this.diagChunks++;
        if (!this.isActive || !this.vad) {
            // Stage 1 drop. Reached when start() was never called, or when
            // spawnWorker failed and tore the instance back down — the audio
            // is captured and then discarded here, with no other signal.
            this.diagDroppedInactive++;
            this.diagReport();
            return;
        }
        const f32 = resampleToF32(chunk, this.inputSampleRate);
        const rms = LocalWhisperSTT.rmsOf(f32);
        if (rms > this.diagPeakRms) this.diagPeakRms = rms;
        const segs = this.vad.push(f32);
        this.diagSegments += segs.length;
        this.diagReport();
        segs.forEach(s => this.dispatchFinal(s.samples));

        // Soft-commit: if a segment has grown past MAX_SEGMENT_MS, force a
        // final pass and start a new (tail-keep) segment. The softCommit
        // bumps the segment id, so the boundary check below picks it up.
        const open = this.vad.peekOpenSegment();
        if (open && open.durationMs >= LocalWhisperSTT.MAX_SEGMENT_MS) {
            const committed = this.vad.softCommit();
            if (committed) this.dispatchFinal(committed.samples);
        }

        // Telemetry: re-stamp segmentOpenedAt whenever the open VAD segment
        // is a different one than we last tracked. ID-based detection
        // correctly handles open+close-in-one-push (two new segments seen
        // within a single write) and close+open-in-one-push (id rises but
        // isInSpeech stays true).
        if (this.vad.isInSpeech()) {
            const id = this.vad.currentSegmentId();
            if (id !== this.trackedSegmentId) {
                this.trackedSegmentId = id;
                this.segmentOpenedAt = performance.now();
                this.firstPartialEmittedForSegment = 0;
            }
        }

        // Reset gap-flush timer.
        if (this.gapFlushTimer) clearTimeout(this.gapFlushTimer);
        this.gapFlushTimer = setTimeout(() => {
            this.gapFlushTimer = null;
            if (this.isActive && this.vad) {
                const pending = this.vad.flush();
                pending.forEach(s => this.dispatchFinal(s.samples));
            }
        }, LocalWhisperSTT.GAP_FLUSH_MS);
    }

    finalize(): void {
        if (!this.isActive || !this.vad) return;
        const segs = this.vad.flush();
        segs.forEach(s => this.dispatchFinal(s.samples));
    }

    /* ──────────────── Streaming inference loop ──────────────── */

    private startStreamingLoop(): void {
        if (this.streamingTimer) return;
        this.streamingNextDelayMs = this.streamingIntervalBaseMs;
        this.streamingStallCount = 0;
        this.scheduleNextStreamingTick();
    }

    private scheduleNextStreamingTick(): void {
        if (!this.isActive) return;
        this.streamingTimer = setTimeout(() => {
            this.streamingTimer = null;
            // Wrap tick in try/catch — a throw here (worker disposed mid-post,
            // VAD nulled, etc.) would otherwise leave the chain unscheduled
            // and silently kill all partials for the rest of the session.
            try {
                this.streamingTick();
            } catch (e) {
                console.warn('[LocalWhisperSTT] streamingTick threw, continuing loop:', e);
                // Treat as a stall so the backoff timer kicks in if the
                // throw is persistent (e.g. recurring postMessage error).
                this.recordStreamingStall();
            }
            this.scheduleNextStreamingTick();
        }, this.streamingNextDelayMs);
    }

    private stopStreamingLoop(): void {
        if (this.streamingTimer) {
            clearTimeout(this.streamingTimer);
            this.streamingTimer = null;
        }
        this.clearStreamingWatchdog();
        this.streamingTaskInFlight = false;
        this.streamingTaskId = null;
        this.streamingStallCount = 0;
        this.streamingNextDelayMs = this.streamingIntervalBaseMs;
    }

    private armStreamingWatchdog(): void {
        this.clearStreamingWatchdog();
        this.streamingWatchdogTimer = setTimeout(() => {
            this.streamingWatchdogTimer = null;
            if (!this.streamingTaskInFlight) return;
            console.warn(`[LocalWhisperSTT] Streaming watchdog fired after ${LocalWhisperSTT.STREAMING_WATCHDOG_MS}ms — worker is stuck, force-clearing in-flight task`);
            const stuckTaskId = this.streamingTaskId;
            // The stuck task will never resolve, so release its occupancy slot
            // too — otherwise inFlightTranscribes only ever climbs and the
            // backpressure check would permanently block all future dispatch,
            // converting a slow worker into a silent one.
            this.noteTranscribeResolved(stuckTaskId ?? undefined);
            this.streamingTaskInFlight = false;
            this.streamingTaskId = null;
            this.streamingStallCount = 0;
            this.streamingNextDelayMs = this.streamingIntervalBaseMs;
            this.emit('error', new Error(
                `Local Whisper streaming task ${stuckTaskId ?? '?'} did not return within ${LocalWhisperSTT.STREAMING_WATCHDOG_MS}ms — worker likely stuck, unblocking next tick.`
            ));
        }, LocalWhisperSTT.STREAMING_WATCHDOG_MS);
    }

    private clearStreamingWatchdog(): void {
        if (this.streamingWatchdogTimer) {
            clearTimeout(this.streamingWatchdogTimer);
            this.streamingWatchdogTimer = null;
        }
    }

    private streamingTick(): void {
        if (!this.isActive || !this.vad || !this.workerReady || !this.worker) {
            this.recordStreamingStall();
            return;
        }
        // Cheap early-return: skip the peekOpenSegment allocation when the
        // VAD isn't currently in a speech segment.
        if (!this.vad.isInSpeech()) { this.recordStreamingStall(); return; }
        // Don't stack streaming requests — wait for the previous one to finish.
        if (this.streamingTaskInFlight) { this.recordStreamingStall('busy'); return; }
        // Real occupancy check. A streaming pass is a disposable live preview,
        // so when the worker is already busy the right move is to skip this
        // tick entirely rather than lengthen its queue.
        if (this.inFlightTranscribes > 0) { this.recordStreamingStall('busy'); return; }

        const open = this.vad.peekOpenSegment();
        if (!open || open.durationMs < this.streamingMinAudioMs) {
            this.recordStreamingStall();
            return;
        }

        // Successful dispatch — reset backoff to base interval.
        this.streamingStallCount = 0;
        this.streamingNextDelayMs = this.streamingIntervalBaseMs;

        this.streamingTaskInFlight = true;
        this.armStreamingWatchdog();
        // Dispatch through the shared path so a streaming pass is COUNTED as
        // worker occupancy like any other. It posts directly before, which meant
        // the partial's noteTranscribeResolved() decremented a counter this
        // dispatch never incremented — an unbalanced release that quietly
        // under-counted how busy the worker really was.
        this.streamingTaskId = this.sendTranscribe(open.samples, true);
    }

    private recordStreamingStall(reason: 'idle' | 'busy' = 'idle'): void {
        // A BUSY stall is not the same event as an IDLE one, and treating them
        // alike is why live partials stopped appearing entirely.
        //
        // During continuous speech the worker is nearly always mid-final, so
        // every streaming tick hit the occupancy check and counted as a stall.
        // Three of those doubled the interval, then doubled again, up to 12
        // seconds — and LocalAgreement-2 needs TWO passes over a segment before
        // it will emit anything, which at that interval never happens. The
        // telemetry says it plainly:
        //
        //   [LocalWhisperSTT/whisper-tiny:mic] latency · first-partial: n=0
        //     · final: n=10 p50=1939ms p95=4211ms
        //
        // Not one partial in a whole session, so every word waited for its
        // final. Backing off is right when there is nothing to transcribe; it is
        // exactly wrong when there is a queue, because the queue is about to
        // clear and we want the next tick ready for it.
        if (reason === 'busy') {
            this.streamingNextDelayMs = this.streamingIntervalBaseMs;
            return;
        }
        this.streamingStallCount++;
        // After 3 consecutive stalls, exponentially back off so we stop
        // spinning while the worker is processing a slow model. Reset only
        // happens on a real dispatch.
        if (this.streamingStallCount >= 3) {
            this.streamingNextDelayMs = Math.min(
                LocalWhisperSTT.STREAMING_INTERVAL_MAX_MS,
                this.streamingNextDelayMs * 2
            );
        }
    }

    /**
     * LocalAgreement-2: commit the longest common prefix between the previous
     * partial and this one. The first partial of a segment seeds the
     * baseline (no emit — agreement requires two passes). Subsequent passes
     * emit only the *new* committed text as an interim transcript.
     */
    private handleStreamingPartial(text: string): void {
        this.clearStreamingWatchdog();
        this.streamingTaskInFlight = false;
        // Worker just became free → recover from any backoff state so the
        // next dispatch fires at the base interval instead of waiting out
        // the doubled delay scheduled while the worker was busy.
        this.streamingStallCount = 0;
        this.streamingNextDelayMs = this.streamingIntervalBaseMs;

        const cleaned = filterHallucination(text);
        if (!cleaned) return;

        // Streaming-class models (Moonshine) produce stable, deterministic
        // output — emit each partial directly. Skipping LA-2's two-pass
        // confirmation cuts an entire tick of latency (~750ms) off the
        // first-text time. The trade-off is occasional flicker on the last
        // word as the model refines, but partial transcripts already carry
        // confidence=0.7 to signal "may change" to consumers.
        if (this.skipAgreement) {
            // Skip duplicate emits when the model produces identical text
            // for consecutive ticks (stable utterance, no new audio).
            if (cleaned !== this.lastEmittedText) {
                this.lastEmittedText = cleaned;
                this.recordFirstPartialLatencyOnce();
                this.emit('transcript', {
                    text: cleaned.trim(),
                    isFinal: false,
                    confidence: 0.7,
                });
            }
            return;
        }

        // LocalAgreement-2 path (Whisper / Distil-Whisper): need two
        // overlapping passes to converge on a stable committed prefix.
        if (this.lastPartialText === '') {
            this.lastPartialText = cleaned;
            return;
        }

        const agreed = this.longestCommonPrefix(this.lastPartialText, cleaned);
        this.lastPartialText = cleaned;

        if (agreed.length > this.lastEmittedText.length) {
            this.lastEmittedText = agreed;
            this.recordFirstPartialLatencyOnce();
            this.emit('transcript', {
                text: this.lastEmittedText.trim(),
                isFinal: false,
                confidence: 0.7,
            });
        }
    }

    private recordFirstPartialLatencyOnce(): void {
        if (this.segmentOpenedAt > 0 && this.firstPartialEmittedForSegment !== this.trackedSegmentId) {
            const dt = performance.now() - this.segmentOpenedAt;
            if (dt > 0 && dt < LocalWhisperSTT.LATENCY_MAX_MS) {
                this.recordLatency(this.firstPartialLatencies, dt);
            }
            this.firstPartialEmittedForSegment = this.trackedSegmentId;
        }
    }

    /* ──────────────── Latency telemetry helpers ──────────────── */

    private recordLatency(arr: number[], ms: number): void {
        arr.push(ms);
        if (arr.length > LocalWhisperSTT.LATENCY_WINDOW) arr.shift();
        this.latencyLogCounter++;
        if (this.latencyLogCounter >= LocalWhisperSTT.LATENCY_LOG_EVERY) {
            this.latencyLogCounter = 0;
            this.logLatencySummary();
        }
    }

    private percentile(sorted: number[], p: number): number {
        if (sorted.length === 0) return 0;
        const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
        return Math.round(sorted[idx]);
    }

    private logLatencySummary(): void {
        const fp = [...this.firstPartialLatencies].sort((a, b) => a - b);
        const fn = [...this.finalLatencies].sort((a, b) => a - b);
        const fmt = (s: number[]) => s.length === 0
            ? 'n=0'
            : `n=${s.length} p50=${this.percentile(s, 50)}ms p95=${this.percentile(s, 95)}ms p99=${this.percentile(s, 99)}ms`;
        const channelTag = this.channelLabel ? `:${this.channelLabel}` : '';
        console.log(`[LocalWhisperSTT/${this.modelId.split('/').pop()}${channelTag}] latency · first-partial: ${fmt(fp)} · final: ${fmt(fn)}`);
    }

    /** Snapshot for UI / IPC. */
    public getLatencyStats(): { firstPartial: { count: number; p50: number; p95: number; p99: number }; final: { count: number; p50: number; p95: number; p99: number } } {
        const fp = [...this.firstPartialLatencies].sort((a, b) => a - b);
        const fn = [...this.finalLatencies].sort((a, b) => a - b);
        return {
            firstPartial: { count: fp.length, p50: this.percentile(fp, 50), p95: this.percentile(fp, 95), p99: this.percentile(fp, 99) },
            final:        { count: fn.length, p50: this.percentile(fn, 50), p95: this.percentile(fn, 95), p99: this.percentile(fn, 99) },
        };
    }

    private longestCommonPrefix(a: string, b: string): string {
        if (!a || !b) return '';
        const len = Math.min(a.length, b.length);
        let i = 0;
        while (i < len && a[i] === b[i]) i++;
        // Snap back to a word boundary ONLY when we've split mid-word — i.e.
        // both sides of position i are non-whitespace. Without this guard the
        // snap-back walked through the entire prefix and produced ''.
        if (i < a.length && /\S/.test(a[i]) && i > 0 && /\S/.test(a[i - 1])) {
            while (i > 0 && /\S/.test(a[i - 1])) i--;
        }
        return a.slice(0, i);
    }

    private resetAgreementState(): void {
        this.lastPartialText = '';
        this.lastEmittedText = '';
        // Invalidate any in-flight streaming task so its late `partial`
        // response is dropped by the taskId guard below instead of mutating
        // the next segment's agreement baseline.
        this.clearStreamingWatchdog();
        this.streamingTaskId = null;
    }

    /* ──────────────── Final segment dispatch ──────────────── */

    private dispatchFinal(audio: Float32Array): void {
        if (!this.worker) {
            // Stage 3 drop. VAD produced a real segment but there is no worker
            // to transcribe it — the segment is lost silently.
            this.diagDroppedNoWorker++;
            return;
        }
        this.diagDispatched++;

        // A final pass closes the streaming window — clear agreement state so
        // the next segment starts clean.
        this.resetAgreementState();
        this.clearStreamingWatchdog();
        this.streamingTaskInFlight = false;

        if (!this.workerReady) {
            const MAX_PENDING = 500;
            if (this.pendingAudio.length < MAX_PENDING) {
                this.pendingAudio.push(audio.slice());
            } else {
                console.warn('[LocalWhisperSTT] Pending queue full — dropping oldest segment');
                this.pendingAudio.shift();
                this.pendingAudio.push(audio.slice());
            }
            return;
        }

        // Count only outstanding FINALS. Now that a streaming pass occupies a
        // slot too, measuring total occupancy here would let a disposable live
        // preview push a real phrase off the queue — trading the transcript for
        // the preview of it, which is exactly backwards. Task ids carry their
        // kind ('t' final, 's' streaming) and the watchdog map holds precisely
        // the unresolved ones.
        const outstandingFinals = [...this.taskWatchdogs.keys()]
            .filter(id => id.startsWith('t')).length;
        if (outstandingFinals >= LocalWhisperSTT.MAX_INFLIGHT_TRANSCRIBES) {
            // Dropping a final loses a phrase, which is bad — but queueing it
            // behind an already-saturated worker loses it anyway (it would
            // surface tens of seconds late) AND delays every phrase after it.
            // Fail visibly on the phrase rather than invisibly on the session.
            // COALESCE INSTEAD OF DROP.
            //
            // Whisper's encoder has a fixed 30-second receptive field: every
            // clip is padded to 30s before it is encoded, so a 0.8s phrase
            // costs the SAME encoder pass as a 25s one. Measured on a 6-core
            // i5-11260H with whisper-small, single channel, 10 threads:
            //
            //   "Проверка."                    8395ms  (10.0x realtime)
            //   "Только что-то выразилось..."  6120ms  ( 2.7x realtime)
            //
            // Nearly the same wall time for very different amounts of speech —
            // that flat floor is the padding, not the words.
            //
            // Dropping a phrase therefore saves nothing structural: the next
            // phrase still pays a whole encoder pass. Appending it to the one
            // already waiting costs NOTHING EXTRA — same single pass — and
            // keeps the words. Four drops became one combined transcription.
            //
            // Capped below the 30s window so the merged clip never exceeds what
            // Whisper can encode in one go.
            this.coalesceFinal(audio);
            return;
        }
        if (this.isDrainingFinals) {
            this.drainingFinalsInFlight++;
        }
        this.sendTranscribe(audio, false);
    }

    /**
     * Audio that arrived while the worker was saturated, held for the next free
     * slot instead of being thrown away.
     *
     * Whisper pads every clip to a 30-second window before encoding, so the
     * encoder cost is FLAT regardless of how much speech the clip contains.
     * That makes dropping a phrase a pure loss — the work saved is zero,
     * because the next phrase pays the same fixed pass anyway. Merging the
     * waiting phrases into one clip transcribes all of them for the price of
     * the one pass that was going to happen regardless.
     */
    private coalesceBuffer: Float32Array[] = [];
    private coalesceSamples = 0;
    /** 25s at 16kHz — comfortably inside Whisper's 30s window. */
    private static readonly MAX_COALESCED_SAMPLES = 25 * 16000;

    private coalesceFinal(audio: Float32Array): void {
        const tag = this.channelLabel ? `:${this.channelLabel}` : '';
        this.coalesceBuffer.push(audio.slice());
        this.coalesceSamples += audio.length;

        // Past the window there is genuinely nothing to do but let the oldest
        // audio go — a clip longer than 30s cannot be encoded in one pass, and
        // splitting it would reintroduce the per-pass cost this avoids.
        while (this.coalesceSamples > LocalWhisperSTT.MAX_COALESCED_SAMPLES && this.coalesceBuffer.length > 1) {
            const dropped = this.coalesceBuffer.shift();
            this.coalesceSamples -= dropped?.length ?? 0;
            this.diagDroppedBackpressure++;
            console.warn(
                `[LocalWhisperSTT${tag}] merged backlog exceeded ` +
                `${LocalWhisperSTT.MAX_COALESCED_SAMPLES / 16000}s — dropping the oldest audio in it`,
            );
            this.reportDegraded(
                `${this.shortModelName()} is far enough behind that speech is being lost. ` +
                `Choose a smaller model in Settings → Audio, or enable GPU acceleration.`,
            );
        }
        console.log(
            `[LocalWhisperSTT${tag}] worker busy — holding ${(this.coalesceSamples / 16000).toFixed(1)}s ` +
            `of speech in ${this.coalesceBuffer.length} segment(s) to transcribe together ` +
            `(one 30s encoder pass covers all of it)`,
        );
    }

    /** Send the held audio as ONE clip, if a slot is free and there is any. */
    private flushCoalesced(): void {
        if (this.coalesceBuffer.length === 0 || !this.worker || !this.workerReady) return;
        const outstandingFinals = [...this.taskWatchdogs.keys()]
            .filter(id => id.startsWith('t')).length;
        if (outstandingFinals >= LocalWhisperSTT.MAX_INFLIGHT_TRANSCRIBES) return;

        const merged = new Float32Array(this.coalesceSamples);
        let offset = 0;
        for (const chunk of this.coalesceBuffer) { merged.set(chunk, offset); offset += chunk.length; }
        const segments = this.coalesceBuffer.length;
        this.coalesceBuffer = [];
        this.coalesceSamples = 0;

        const tag = this.channelLabel ? `:${this.channelLabel}` : '';
        console.log(
            `[LocalWhisperSTT${tag}] transcribing ${segments} held segment(s) as one ` +
            `${(merged.length / 16000).toFixed(1)}s clip`,
        );
        this.diagDispatched++;
        if (this.isDrainingFinals) this.drainingFinalsInFlight++;
        this.sendTranscribe(merged, false);
    }

    private sendTranscribe(audio: Float32Array, streaming: boolean): string | null {
        if (!this.worker) return null;
        const taskId = `${streaming ? 's' : 't'}${++this.taskCounter}`;
        const copy = audio.slice();
        this.inFlightTranscribes++;
        this.armTaskWatchdog(taskId, copy.length / 16000);
        this.worker.postMessage(
            { type: 'transcribe', taskId, audio: copy, language: this.language, streaming },
            [copy.buffer]
        );
        return taskId;
    }

    /**
     * Per-task watchdog covering FINALS as well as streaming passes.
     *
     * Only streaming tasks used to be watched. Finals were not, and a slow model
     * turned that gap into total silence — from a packaged Windows build running
     * whisper-large-v3-turbo on CPU:
     *
     *   transcribe START task=t1 samples=25440 (1.59s)   <- never returned
     *   ... inFlight=2 droppedBackpressure=1
     *   ... inFlight=2 droppedBackpressure=2   (every later phrase dropped)
     *
     * Two finals went out, neither came back, inFlightTranscribes stayed pinned
     * at the backpressure limit, and the channel refused every phrase for the
     * rest of the meeting. The worker cannot report this itself — heavy ONNX
     * inference starves its timers (see whisperWorker's watchdog caveat) — so the
     * host, whose event loop stays free, has to be the one holding the stopwatch.
     *
     * On expiry the slot is released so the channel keeps accepting speech. A
     * result that arrives after this point is still delivered; the taskId is kept
     * in timedOutTasks only so its late arrival doesn't decrement the counter
     * twice and over-admit work into a worker that is demonstrably still busy.
     */
    private armTaskWatchdog(taskId: string, audioSeconds: number): void {
        // Scale with the audio: a 14s segment legitimately takes longer than a
        // 1s one. Past the cap the result is too late to be worth anything.
        const budget = Math.min(
            LocalWhisperSTT.TASK_WATCHDOG_MAX_MS,
            Math.max(LocalWhisperSTT.TASK_WATCHDOG_MIN_MS, audioSeconds * 1000 * 8),
        );
        const startedAt = Date.now();
        const timer = setTimeout(() => {
            this.taskWatchdogs.delete(taskId);
            // A task still outstanding when the session ends is expected, not a
            // fault — don't accuse the model of being slow on the way out.
            if (!this.isActive && !this.isDrainingFinals) {
                this.noteTranscribeResolved(taskId);
                return;
            }
            const tag = this.channelLabel ? `:${this.channelLabel}` : '';
            console.warn(
                `[LocalWhisperSTT${tag}] task ${taskId} has not returned after ` +
                `${Date.now() - startedAt}ms for ${audioSeconds.toFixed(2)}s of audio ` +
                `(model=${this.modelId}) — releasing its queue slot so later speech is ` +
                `still accepted. This model is too slow for real-time on this machine.`,
            );
            this.reportDegraded(
                `${this.shortModelName()} did not finish transcribing a ${audioSeconds.toFixed(1)}s ` +
                `phrase within ${Math.round(budget / 1000)}s. Choose a smaller model in Settings → Audio.`,
            );
            this.noteTranscribeResolved(taskId);
        }, budget);
        if (typeof timer.unref === 'function') timer.unref();
        this.taskWatchdogs.set(taskId, timer);
    }

    private shortModelName(): string {
        return getModelDisplayName(this.modelId);
    }

    /**
     * Say — before the meeting, not 25 seconds into it — when this model cannot
     * keep up on this machine.
     *
     * Whisper Large v3 Turbo with no GPU is the case that motivated this. On a
     * 6-core i5-11260H it did not return a transcript for a 1.9s phrase within
     * 25 seconds and pushed the process to 6.8GB resident, which then starved
     * the other channel's 39MB Tiny model into the same timeout. Everything
     * about that failure is predictable at start(): the model's speed tier and
     * whether a GPU was found are both already known.
     */
    private warnIfModelTooSlowForThisMachine(): void {
        if (!isTooSlowForCpuRealtime(this.modelId)) return;
        let onGpu = false;
        try {
            const { getResolvedGpuDevice } = require('./whisper/gpuProbe');
            onGpu = getResolvedGpuDevice()?.deviceId !== null && getResolvedGpuDevice() !== null;
        } catch { /* no probe ⇒ CPU */ }
        if (onGpu) return;
        const tag = this.channelLabel ? `:${this.channelLabel}` : '';
        const alternatives = getCpuRealtimeModelNames(this.language !== 'auto' && !this.language.startsWith('english'));
        console.warn(
            `[LocalWhisperSTT${tag}] ${this.shortModelName()} is a large model and no GPU was ` +
            `available, so it will run on the CPU. Expect it to fall behind live speech and skip ` +
            `phrases. Faster choices for this machine: ${alternatives.join(', ')}.`,
        );
        this.reportDegraded(
            `${this.shortModelName()} is too large to keep up on this computer's CPU. ` +
            `Switch to ${alternatives[0] ?? 'a smaller model'} in Settings → Audio.`,
        );
    }

    /**
     * Tell the user their transcription is degraded, through the same channel
     * every other STT provider uses for failures.
     *
     * This gap is the difference between "Natively is broken" and "this model
     * is too slow for this machine". A cloud provider that stops working emits
     * an error and the meeting UI shows it. Local Whisper starving on its own
     * queue emitted NOTHING — the log filled with dropped segments while the
     * overlay showed a normal, working meeting that simply never produced a
     * word. Silence is the worst possible way to report a fault, because the
     * user cannot tell it apart from having nothing to say.
     *
     * Rate-limited: one report per interval per channel. Backpressure drops
     * arrive several times a minute and each one is the same news.
     */
    private reportDegraded(message: string): void {
        const now = Date.now();
        if (now - this.lastDegradedReportAt < LocalWhisperSTT.DEGRADED_REPORT_INTERVAL_MS) return;
        this.lastDegradedReportAt = now;
        this.emit('error', new Error(message));
    }

    private lastDegradedReportAt = 0;
    private static readonly DEGRADED_REPORT_INTERVAL_MS = 20000;

    private taskWatchdogs = new Map<string, ReturnType<typeof setTimeout>>();
    /**
     * Task ids whose queue slot has already been given back, so whichever of
     * result / error / watchdog arrives second is a no-op. Without this, a task
     * that times out and then answers anyway is counted down twice and the
     * channel over-admits work into a worker that is demonstrably still busy.
     * Bounded — only the recent tail can produce a duplicate.
     */
    private releasedTasks = new Set<string>();
    private static readonly RELEASED_TASK_MEMORY = 64;
    private static readonly TASK_WATCHDOG_MIN_MS = 25000;
    private static readonly TASK_WATCHDOG_MAX_MS = 120000;

    /**
     * Number of transcribe messages posted to the worker but not yet resolved.
     *
     * streamingTaskInFlight alone does NOT describe worker occupancy:
     * dispatchFinal CLEARS it and then posts a final, so the streaming loop
     * immediately believes the worker is free and ticks again — while the
     * worker is still busy with both. The worker handles messages serially, so
     * every extra post just lengthens a queue.
     *
     * With inference slower than speech, that queue grows without bound.
     * Observed on a packaged build, three tasks outstanding at once and the
     * oldest starved:
     *
     *   transcribe STILL RUNNING task=t2 elapsed=14477ms
     *   transcribe STILL RUNNING task=s3 elapsed=8729ms
     *   transcribe STILL RUNNING task=t4 elapsed=8507ms
     *   transcribe STILL RUNNING task=s3 elapsed=188972ms   <- 3+ minutes
     *
     * A result 189 seconds late is worthless, and producing it starves every
     * later utterance. This counter is the real occupancy signal used to apply
     * backpressure. It is never allowed below zero: a watchdog force-clear or a
     * late message must not corrupt it into permanently blocking dispatch.
     */
    private inFlightTranscribes = 0;

    /**
     * Outstanding finals beyond which new ones are dropped rather than queued.
     * Finals carry the actual transcript so they are not dropped lightly, but a
     * third queued final lands ~30s late at observed speeds — long after the
     * words matter — while delaying everything behind it. On a healthy setup
     * (GPU, or a small model) inference outruns speech and this never trips.
     */
    private static readonly MAX_INFLIGHT_TRANSCRIBES = 2;

    /** Give a task's queue slot back, exactly once per task. */
    private noteTranscribeResolved(taskId?: string): void {
        if (taskId) {
            const timer = this.taskWatchdogs.get(taskId);
            if (timer) {
                clearTimeout(timer);
                this.taskWatchdogs.delete(taskId);
            }
            if (this.releasedTasks.has(taskId)) return;
            this.releasedTasks.add(taskId);
            if (this.releasedTasks.size > LocalWhisperSTT.RELEASED_TASK_MEMORY) {
                const oldest = this.releasedTasks.values().next().value;
                if (oldest !== undefined) this.releasedTasks.delete(oldest);
            }
        }
        this.inFlightTranscribes = Math.max(0, this.inFlightTranscribes - 1);
        // A slot just freed — send whatever was held rather than leaving it to
        // expire. Safe to call unconditionally; it returns immediately when
        // there is nothing held or no room.
        this.flushCoalesced();
    }

    private clearAllTaskWatchdogs(): void {
        for (const timer of this.taskWatchdogs.values()) clearTimeout(timer);
        this.taskWatchdogs.clear();
        this.releasedTasks.clear();
    }

    /* ──────────────── Worker lifecycle ──────────────── */

    private async spawnWorker(): Promise<void> {
        const warm = modelPreloader.takeWarmWorker(this.modelId);
        if (warm) {
            console.log(`[LocalWhisperSTT] Using preloaded warm worker for ${this.modelId}`);
            this.worker = warm;
            this.markWorkerLive();
            this.workerReady = true;
            // Inherit the slot release the preloader acquired. Both preloader
            // and our local listeners will call this — it's a no-op the
            // second time.
            this.slotRelease = (warm as any).__slotRelease ?? null;
            this.attachWorkerListeners();
            this.flushPending();
            return;
        }

        // Cold path. Acquire the shared ONNX slot at HIGH priority — Whisper
        // is latency-critical (~750ms real-time streaming) and would deadlock
        // behind a queued embedding batch.
        if (!hasEnoughMemoryForOnnxSession()) {
            const heapGB = (process.memoryUsage().heapUsed / 1024 ** 3).toFixed(1);
            throw new Error(
                `[LocalWhisperSTT] insufficient available memory (<${getMinFreeGBForOnnxSession()}GB free) — ` +
                `Whisper init refused for ${this.modelId} (available=${getAvailableMemoryGB().toFixed(1)}GB, ` +
                `this process heap=${heapGB}GB). A model download in progress is the usual cause: it holds the ` +
                `whole model in memory while fetching, so nothing transcribes until it finishes or is paused.`,
            );
        }

        // A SECOND concurrent session is a second FULL copy of the model in
        // this process — worker_threads share the process heap, so mic +
        // system audio means 2x the weights resident, not a shared cache.
        //
        // The flat floor above cannot catch this: it is checked BEFORE the
        // load, when memory still looks fine, and the exhaustion happens
        // DURING the load. Observed on a 16GB Windows machine with
        // whisper-large-v3-turbo: RSS climbed to 7.9GB with 811MB free, the
        // second worker never reached `ready`, and — because both workers
        // share the process — the FIRST (already-loaded, healthy) worker could
        // no longer complete an inference either. Result: audio dispatched,
        // never transcribed, no error anywhere.
        //
        // So an additional session must be admitted against the footprint it
        // will actually add. ONNX resident size runs well above the on-disk
        // size (fp32 weights plus arenas); 2x is a deliberately conservative
        // estimate — under-admitting costs one channel, over-admitting costs
        // BOTH channels plus responsiveness.
        if (LocalWhisperSTT.liveWorkers > 0) {
            const modelGB = (getModelSizeBytes(this.modelId) || 0) / 1024 ** 3;
            const requiredGB = getMinFreeGBForOnnxSession() + modelGB * 2;
            const availableGB = getAvailableMemoryGB();
            if (modelGB > 0 && availableGB < requiredGB) {
                throw new Error(
                    `[LocalWhisperSTT] refusing a 2nd concurrent Whisper session for ${this.modelId}: ` +
                    `available=${availableGB.toFixed(1)}GB < required=${requiredGB.toFixed(1)}GB ` +
                    `(model=${modelGB.toFixed(1)}GB x2 + ${getMinFreeGBForOnnxSession()}GB floor). ` +
                    `Loading it would stall this channel AND the one already running. ` +
                    `Pick a smaller local model, or set per-channel models in Settings.`,
                );
            }
        }

        this.slotRelease = await acquireOnnxSlot('high');

        console.log(`[LocalWhisperSTT] Cold-starting worker for ${this.modelId}`);
        const workerPath = resolveWhisperWorkerPath();
        writeLoadSentinel(this.modelId);
        this.worker = new Worker(workerPath);
        this.markWorkerLive();
        this.attachWorkerListeners();
        this.worker.postMessage(buildWorkerInitMessage(this.modelId));
    }

    private attachWorkerListeners(): void {
        if (!this.worker) return;

        this.worker.on('message', (msg: WorkerOutMessage) => {
            // Handled before every other branch — including the isActive gate
            // below — because the most valuable worker logs are the ones from a
            // worker that is loading, stuck, or dying, i.e. exactly when this
            // instance is not yet (or no longer) active.
            if ((msg as { type?: string }).type === 'log') {
                const m = msg as unknown as { level?: string; message?: string };
                const tag = this.channelLabel ? `:${this.channelLabel}` : '';
                const line = `[WhisperWorker${tag}] ${m.message ?? ''}`;
                if (m.level === 'error') console.error(line);
                else if (m.level === 'warn') console.warn(line);
                else console.log(line);
                return;
            }
            if (msg.type === 'ready') {
                clearLoadSentinel(this.modelId);
                this.workerReady = true;
                this.flushPending();
                return;
            }

            // After stop(), allow only the explicitly flushed final segments to
            // return during the 5s drain window; partials and unrelated worker
            // messages remain ignored on a torn-down instance.
            if (!this.isActive && !(this.isDrainingFinals && msg.type === 'result')) return;

            if (msg.type === 'partial') {
                // Drop partials whose segment has already been finalized — the
                // agreement baseline is reset on every final dispatch and the
                // taskId is invalidated, so a late partial would otherwise
                // corrupt the next segment.
                this.noteTranscribeResolved(msg.taskId);
                if (msg.taskId !== this.streamingTaskId) {
                    this.streamingTaskInFlight = false;
                    return;
                }
                this.handleStreamingPartial(msg.text);
            } else if (msg.type === 'result') {
                this.noteTranscribeResolved(msg.taskId);
                const text = filterHallucination(msg.text);
                // A result that arrives and is then filtered to nothing is
                // indistinguishable, from the outside, from a result that never
                // arrived: both end in silence with no error. Log the raw length
                // and the filter's verdict so the two can be told apart.
                const tag = this.channelLabel ? `:${this.channelLabel}` : '';
                console.log(
                    `[LocalWhisperSTT${tag}] result task=${(msg as { taskId?: string }).taskId ?? '?'}` +
                    ` rawChars=${(msg.text ?? '').length} kept=${text ? 'yes' : 'NO (filtered as hallucination)'}` +
                    ` raw="${String(msg.text ?? '').slice(0, 60)}"`,
                );
                if (text) {
                    if (this.segmentOpenedAt > 0) {
                        const dt = performance.now() - this.segmentOpenedAt;
                        if (dt > 0 && dt < LocalWhisperSTT.LATENCY_MAX_MS) {
                            this.recordLatency(this.finalLatencies, dt);
                        }
                    }
                    this.emit('transcript', { text, isFinal: true, confidence: 0.9 });
                }
                // Reset segment timer regardless of emit (silent finals also close
                // the segment). Next write() that opens a fresh VAD segment will
                // re-stamp via the segment-id check.
                this.segmentOpenedAt = 0;
                if (this.isDrainingFinals) {
                    this.drainingFinalsInFlight = Math.max(0, this.drainingFinalsInFlight - 1);
                    if (this.drainingFinalsInFlight === 0 && this.worker) {
                        this.beginWorkerTermination(this.worker);
                    }
                }
            } else if (msg.type === 'error') {
                this.noteTranscribeResolved(msg.taskId);
                console.error('[LocalWhisperSTT] Worker error:', msg.message);
                if (this.isDrainingFinals && msg.taskId?.startsWith('t')) {
                    this.drainingFinalsInFlight = Math.max(0, this.drainingFinalsInFlight - 1);
                    if (this.drainingFinalsInFlight === 0 && this.worker) {
                        this.beginWorkerTermination(this.worker);
                    }
                }
                // If the failed task was the in-flight streaming one, unblock
                // the loop so the next tick can fire.
                if (msg.taskId && msg.taskId === this.streamingTaskId) {
                    this.streamingTaskInFlight = false;
                    this.streamingTaskId = null;
                    // Worker is free again; reset backoff so next tick is prompt.
                    this.streamingStallCount = 0;
                    this.streamingNextDelayMs = this.streamingIntervalBaseMs;
                }
                if (msg.message.includes('Failed to load model')) {
                    const isOnnxSymbolError = msg.message.includes('Symbol not found')
                        || msg.message.includes('__ZNSt3__18to_charsEPcS0_d')
                        || msg.message.includes('libonnxruntime');
                    this.emit('error', new Error(
                        isOnnxSymbolError
                            ? 'Local Whisper is not supported on macOS 12 (Monterey) or earlier. Please upgrade to macOS 13 Ventura or later, or use a cloud STT provider.'
                            : 'Local Whisper model not found. Please download a model in Settings → Audio.'
                    ));
                }
            }
        });

        this.worker.on('error', (err) => {
            // Reset all in-flight streaming state so a dead worker can never
            // permanently pin streamingTaskInFlight=true (which would freeze
            // the loop — symptom: transcription stops after 3-4 questions).
            this.clearStreamingWatchdog();
            this.clearAllTaskWatchdogs();
            this.coalesceBuffer = [];
            this.coalesceSamples = 0;
            this.inFlightTranscribes = 0;
            this.streamingTaskInFlight = false;
            this.streamingTaskId = null;
            // Free the shared ONNX gate slot — Whisper's session is gone.
            if (this.slotRelease) { this.slotRelease(); this.slotRelease = null; }
            this.markWorkerGone();
            this.workerReady = false;
            // Symmetric with the exit handler below: a worker `error` is
            // followed by a non-zero `exit` in node:worker_threads, so the
            // exit handler also calls recordLoadFailure. Calling here too is
            // belt-and-braces for the theoretical error-without-exit case
            // (e.g. a hard native abort that races the parent). Idempotent
            // because recordLoadFailure only sets a map expiry, never clears.
            modelPreloader.recordLoadFailure(this.modelId);
            const isOnnxSymbolError = err.message.includes('Symbol not found')
                || err.message.includes('to_chars')
                || err.message.includes('libonnxruntime');
            if (isOnnxSymbolError) {
                this.emit('error', new Error(
                    'Local Whisper is not supported on macOS 12 (Monterey) or earlier. Please upgrade to macOS 13 Ventura or later, or use a cloud STT provider.'
                ));
            } else {
                this.emit('error', err);
            }
        });

        // 'exit' fires whenever the worker terminates (voluntarily or not),
        // including the 'error' path above. If the worker is gone, the
        // streaming loop must be unblocked — otherwise streamingTaskInFlight
        // stays true and the next tick silently stalls forever.
        this.worker.on('exit', (code) => {
            if (code === 0) {
                clearLoadSentinel(this.modelId);
                return; // clean shutdown
            }
            modelPreloader.recordLoadFailure(this.modelId);
            this.clearStreamingWatchdog();
            this.clearAllTaskWatchdogs();
            this.coalesceBuffer = [];
            this.coalesceSamples = 0;
            this.inFlightTranscribes = 0;
            if (this.slotRelease) { this.slotRelease(); this.slotRelease = null; }
            this.markWorkerGone();
            const hadInFlight = this.streamingTaskInFlight;
            this.streamingTaskInFlight = false;
            this.streamingTaskId = null;
            this.workerReady = false;
            if (hadInFlight) {
                this.emit('error', new Error(
                    `Local Whisper worker exited unexpectedly (code=${code}) — transcription stream has been unblocked.`
                ));
            }
        });
    }

    private flushPending(): void {
        // Push the cached prompt to the worker FIRST so the queued transcribes
        // see the bias on their initial run (worker honors the latest cached
        // prompt for whichever transcribe arrives next).
        this.maybePushPromptToWorker();
        const queued = this.pendingAudio.splice(0);
        queued.forEach(audio => this.sendTranscribe(audio, false));
        if (this.isDrainingFinals && queued.length === 0 && this.drainingFinalsInFlight === 0 && this.worker) {
            this.beginWorkerTermination(this.worker);
        }
    }

    private beginWorkerTermination(w: Worker): void {
        this.worker = null;
        this.workerReady = false;
        this.isDrainingFinals = false;
        this.drainingFinalsInFlight = 0;
        // Free the shared ONNX gate slot on clean shutdown — the session's
        // BFCArena is being torn down with the worker, so the slot can go.
        if (this.slotRelease) { this.slotRelease(); this.slotRelease = null; }
            this.markWorkerGone();
        // Reset the sent-prompt tracker: a future spawnWorker call will get a
        // fresh worker with empty cache, so we must re-push on next ready.
        this.contextPromptSentToWorker = '';
        w.removeAllListeners('message');
        w.removeAllListeners('error');
        if (this.workerTerminateTimer) clearTimeout(this.workerTerminateTimer);
        const t = setTimeout(() => {
            this.workerTerminateTimer = null;
            w.terminate();
        }, 5000);
        // unref so the timer doesn't pin the Node event loop on app quit.
        (t as any).unref?.();
        this.workerTerminateTimer = t;
    }
}
