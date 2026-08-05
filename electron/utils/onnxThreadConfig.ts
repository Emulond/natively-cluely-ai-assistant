// electron/utils/onnxThreadConfig.ts
//
// Shared bounded ONNX Runtime thread-count config + cross-loader concurrency
// gate for every local onnxruntime-node consumer in this app
// (LocalEmbeddingProvider, LocalReranker, IntentClassifier's zero-shot
// worker, Whisper's worker).
//
// WHY THIS EXISTS (2026-07-05 SIGTRAP crash hardening):
// 9/9 real macOS crash reports (~/Library/Logs/DiagnosticReports/Electron-*.ips)
// showed an identical main-thread crash inside onnxruntime::BFCArena::Extend →
// posix_memalign, happening during a live InferenceSession::Run() call, with
// 16-17 ORT-related OS threads alive at crash time. This is consistent with
// multiple ONNX Runtime sessions (Whisper STT + IntentClassifier + a local
// embedding/rerank fallback) racing on native allocator/thread-pool resources
// when several are concurrently active in-process.
//
// A creation-time mutex does NOT help — the crash is inside Run(), not
// Create(). Instead, every loader bounds its OWN session to a small, fixed
// number of intra/inter-op threads via ONNX Runtime SessionOptions. This
// caps the total native thread/memory pressure any single session can
// generate, so even when multiple sessions are concurrently executing the
// aggregate stays low — without fully serializing inference across loaders
// (which would throttle Whisper's ~750ms real-time streaming loop
// unacceptably).
//
// Conservative defaults: 1 intra-op thread (no internal op-level
// parallelism) and 1 inter-op thread (sequential execution mode; these
// models have no independent parallel subgraphs to exploit anyway). This is
// the safest configuration for small/quantized transformer models like
// MiniLM, mobilebert, bge-reranker-base, and Whisper's encoder/decoder —
// none of these benefit meaningfully from multi-threaded intra-op execution
// at these model sizes, so the throughput cost of bounding is minimal while
// the crash-surface reduction is significant.
//
// Overridable via env vars for local experimentation / future retuning
// without a code change.
//
// Layer 2 (2026-07-06): shared cross-loader concurrency semaphore +
// free-memory floor. The original crash forensics showed the issue is NOT
// per-session thread count — it's the aggregate pressure of multiple
// concurrent ONNX sessions on the BFCArena. A global acquire/release gate
// caps how many can be live at once, and a `os.freemem()` floor refuses
// any new session when the system is tight. All four consumers
// (LocalEmbeddingProvider, LocalReranker, IntentClassifier, Whisper) gate
// here before posting `init` to their worker.

import os from 'os';
import { execFileSync } from 'child_process';
import fs from 'fs';

export interface OnnxThreadBounds {
    intraOpNumThreads: number;
    interOpNumThreads: number;
    executionMode: 'sequential' | 'parallel';
    enableCpuMemArena?: boolean;
    enableMemPattern?: boolean;
}

function readIntEnv(name: string, fallback: number): number {
    const raw = process.env[name];
    if (!raw) return fallback;
    const n = Number.parseInt(raw, 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
}

function readBoolEnv(name: string, fallback: boolean): boolean {
    const raw = process.env[name];
    if (!raw) return fallback;
    const normalized = raw.trim().toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
    if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
    return fallback;
}

/**
 * Bounded thread-count session options shared by every local ONNX consumer.
 * Kept as a fresh object per call (session_options is merged/mutated by
 * transformers.js internals — never share one object across sessions).
 */
/**
 * Default intra-op thread count.
 *
 * The single-thread default was introduced as the "conservative half" of a fix
 * for a macOS crash (BFCArena::Extend / posix_memalign on 16GB Apple Silicon
 * with several ONNX consumers live). It was applied to EVERY platform, which
 * pins Whisper inference to one core everywhere.
 *
 * Measured cost, from a packaged Windows build's own telemetry:
 *
 *   [LocalWhisperSTT/whisper-small:mic] latency · final: n=1 p50=12176ms
 *
 * 12.2s to transcribe one segment — roughly 10x slower than the speech being
 * transcribed. Results land after the meeting has ended, so the app looks like
 * it produces no transcript at all, with nothing failing.
 *
 * macOS keeps 1 (the crash it mitigates is real and platform-specific).
 * Elsewhere, scale with the machine but stay bounded: half the cores, capped at
 * 4, so a Whisper session cannot monopolise the box or reintroduce the arena
 * pressure the cap was guarding against. Both remain env-overridable.
 */
/**
 * Default for ORT's CPU memory arena + memory-pattern planning.
 *
 * Both were disabled as part of the same macOS crash mitigation as the
 * single-thread default ("crash forensics point at BFCArena::Extend; standard
 * system allocations are safer inside Electron"). Disabling the arena means
 * every tensor allocation during inference goes to the raw system allocator
 * rather than ORT's pool — for transformer decoding that is thousands of
 * allocations per generated token, and it degrades EVERY local ONNX model, not
 * just Whisper.
 *
 * The evidence that this is not merely theoretical, from one packaged Windows
 * session: a MiniLM embedding — a tiny model that normally completes in tens of
 * milliseconds — hit its ceiling:
 *
 *   [EmbeddingPipeline] embed() timed out after 30000ms for chunk 86 via local
 *
 * macOS keeps the conservative setting; the BFCArena crash it guards against is
 * real and specific to that allocator. Elsewhere the arena goes back on. Both
 * remain env-overridable, so a machine that misbehaves can be pinned back with
 * NATIVELY_ONNX_ENABLE_CPU_MEM_ARENA=0.
 */
function defaultArenaEnabled(): boolean {
    return process.platform !== 'darwin';
}

/**
 * How many Whisper sessions the thread budget has to cover.
 *
 * A meeting runs TWO — microphone and system audio — and each was taking
 * `cores - 2` intra-op threads for itself. On a 6-core/12-thread i5-11260H that
 * is 10 threads twice over: 20 threads fighting for 12 logical processors, with
 * the OS scheduler preempting ONNX's own spin-waits. Measured on that machine,
 * whisper-tiny.en — a 39MB model — took 2.9 seconds to transcribe 1.89 seconds
 * of speech while large-v3-turbo ran beside it. Tiny should be many times
 * faster than real time.
 *
 * Oversubscription does not make transcription faster; it makes both channels
 * slower than either would be alone. The budget is therefore machine-wide and
 * split, not per-session.
 */
const CONCURRENT_WHISPER_SESSIONS = 2;

function defaultIntraOpThreads(concurrentSessions = CONCURRENT_WHISPER_SESSIONS): number {
    if (process.platform === 'darwin') return 1;
    const cores = os.cpus?.()?.length ?? 1;
    // Speech transcription is the latency-critical foreground task on this
    // machine — there is nothing to save the cores for. The earlier half-cores
    // cap was cautious to the point of being the bottleneck: on a 12-thread
    // i5-11260H it used 4.
    //
    // Leave 2 threads for the UI, audio capture and the rest of Electron, and
    // take the rest. Capped at 8: past that, ONNX's own coordination overhead
    // starts eating the gain on a consumer CPU, and two channels can be live at
    // once (mic + system audio), each with its own session.
    // os.cpus() reports LOGICAL processors (12 on a 6-core/12-thread i5-11260H),
    // and ONNX scales on physical cores rather than SMT siblings — so leaving
    // two aside still hands inference more threads than the machine has real
    // cores. No upper cap: transcription is the foreground task the user is
    // waiting on, and NATIVELY_ONNX_INTRA_OP_THREADS pins it if a machine
    // prefers otherwise.
    //
    // The budget is then SPLIT across the concurrent sessions rather than handed
    // to each in full — see CONCURRENT_WHISPER_SESSIONS. On the i5-11260H that
    // is 5 threads per channel, 10 in total, instead of 20 threads thrashing 12
    // logical processors.
    const budget = Math.max(1, cores - 2);
    // Divided by however many sessions will REALLY be live. Hardcoding 2 meant a
    // user who turned one channel off still ran the survivor on half the cores,
    // paying for a session that no longer exists.
    const sessions = Math.max(1, Math.floor(concurrentSessions) || 1);
    // Never below 2 per session on a multi-core machine: a single-threaded
    // encoder is the bottleneck this whole function exists to remove.
    return Math.max(cores > 2 ? 2 : 1, Math.floor(budget / sessions));
}

export function getBoundedOnnxSessionOptions(concurrentSessions?: number): OnnxThreadBounds {
    return {
        intraOpNumThreads: readIntEnv('NATIVELY_ONNX_INTRA_OP_THREADS', defaultIntraOpThreads(concurrentSessions)),
        interOpNumThreads: readIntEnv('NATIVELY_ONNX_INTER_OP_THREADS', 1),
        executionMode: 'sequential',
        // Disable ORT's persistent BFCArena/memory-pattern reuse by default.
        // The crash forensics above point at BFCArena::Extend; standard system
        // allocations are safer inside Electron. Env vars keep this reversible
        // for perf experiments without shipping a new build.
        enableCpuMemArena: readBoolEnv('NATIVELY_ONNX_ENABLE_CPU_MEM_ARENA', defaultArenaEnabled()),
        enableMemPattern: readBoolEnv('NATIVELY_ONNX_ENABLE_MEM_PATTERN', defaultArenaEnabled()),
    };
}

/**
 * Session options for a session that will run on the DirectML execution
 * provider.
 *
 * DirectML is not a drop-in: ONNX Runtime documents two hard requirements, and
 * violating either makes session creation fail —
 *
 *   "The DirectML execution provider does not support the use of memory
 *    pattern optimizations or parallel execution in onnxruntime. When supplying
 *    session options during InferenceSession creation, these options must be
 *    disabled or an error will be returned."
 *   — onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
 *
 * That is why the two earlier attempts at a GPU path took the whole app down at
 * startup rather than falling back: the shared session options carry
 * enableMemPattern (on by default off macOS, for CPU throughput) straight into
 * a DirectML session, ORT rejects the combination inside native code, and the
 * failure arrives as a process abort that no JS try/catch can intercept.
 *
 * enableCpuMemArena goes off too. It is not prohibited, but a DirectML session
 * allocates on the GPU; a CPU arena only adds host memory to a process already
 * holding two Whisper models resident.
 *
 * The thread counts are left alone. They are ignored for GPU-assigned nodes and
 * still apply to whatever falls back to the CPU provider.
 */
export function getDirectMLSessionOptions(concurrentSessions?: number): OnnxThreadBounds {
    return {
        ...getBoundedOnnxSessionOptions(concurrentSessions),
        executionMode: 'sequential',
        enableMemPattern: false,
        enableCpuMemArena: false,
    };
}

// ── Cross-loader concurrency gate ──────────────────────────────────────────
//
// A small async semaphore + memory floor shared by every local ONNX
// consumer. Acquired main-side BEFORE posting `init` to a worker —
// worker_threads have separate JS heaps so the in-memory counter must live
// in the main process. Default cap: 2 concurrent sessions (Whisper + one
// other). On a 16GB MacBook Air with 4 native ONNX consumers live
// simultaneously, the BFC arena can grow into the multi-hundred-MB range
// and `posix_memalign` traps. The gate is the structural half of the fix
// for that crash surface; per-session `getBoundedOnnxSessionOptions()`
// (intra/inter-op = 1) is the conservative half.
//
// Refusal policy: the slot release function is async-safe; calling it more
// than once is a no-op. Acquisition fails OPEN if `os.freemem()` itself
// throws (rare sandboxed Linux configs) — the failure case is just
// measurement, not a real signal of trouble.

export type OnnxSlotPriority = 'normal' | 'high';

let inFlightNormal = 0;
let inFlightHigh = 0;
const waitersNormal: Array<() => void> = [];
const waitersHigh: Array<() => void> = [];

function readMaxConcurrent(): number {
    return readIntEnv('NATIVELY_ONNX_MAX_CONCURRENT_SESSIONS', 2);
}

function readMinFreeGB(): number {
    const raw = process.env.NATIVELY_ONNX_MIN_FREE_GB;
    if (!raw) return 2.0;
    const n = Number.parseFloat(raw);
    return Number.isFinite(n) && n >= 0 ? n : 2.0;
}

// Whisper runs one session per audio channel — microphone and system audio —
// so at most two high-priority sessions exist, both for the length of a meeting.
//
// Long-lived 'normal' consumers permanently occupy the shared cap: once loaded,
// IntentClassifier (and LocalReranker) stash their release on the instance and
// hold the slot for the process lifetime. With the default cap of 2, a loaded
// IntentClassifier plus the FIRST Whisper channel fills it, and the SECOND
// Whisper channel blocks on acquire for the entire meeting. In practice that
// second channel is the microphone: system audio calls start() first and takes
// the preloader's warm worker, so the mic is the one left queued. Its audio is
// then discarded in dispatchFinal (no worker) — captured audio, no transcript,
// and no error anywhere to explain it.
//
// 'high' priority already blocks NEW normal acquisitions, but it deliberately
// does not preempt a RUNNING one, so that alone cannot dislodge a session held
// for the process lifetime. Give high-priority acquisitions dedicated headroom
// instead, bounded by the number of audio channels so concurrency stays capped.
const HIGH_PRIORITY_RESERVED_SLOTS = 2;

function canAcquireNow(priority: OnnxSlotPriority): boolean {
    const cap = readMaxConcurrent();
    if (priority === 'high') {
        if (inFlightHigh < HIGH_PRIORITY_RESERVED_SLOTS) return true;
        return inFlightNormal + inFlightHigh < cap;
    }
    // Normal priority: only acquire when there are no high-priority waiters
    // queued (so Whisper can grab the next slot promptly).
    if (waitersHigh.length > 0) return false;
    return inFlightNormal + inFlightHigh < cap;
}

/**
 * Acquire a shared ONNX session slot. Returns a release function the caller
 * MUST call when the session is torn down (typically in worker `error`/`exit`
 * handlers). Blocks until a slot is available; NEVER rejects.
 *
 * Priority 'high' is for latency-critical consumers (Whisper) — it acquires
 * ahead of queued normal-priority waiters but does NOT preempt a running
 * session. If the cap is exhausted, high-priority waiters block normal-priority
 * acquisitions so Whisper can take the next free slot promptly.
 */
export async function acquireOnnxSlot(priority: OnnxSlotPriority = 'normal'): Promise<() => void> {
    const queue = priority === 'high' ? waitersHigh : waitersNormal;
    // Diagnostics: a caller blocked here is invisible from the outside — it has
    // not logged yet and never errors, so a starved Whisper instance looks
    // identical to one that was never started. Record how long each acquisition
    // actually waits, and against whom.
    const waitStartedAt = Date.now();
    let didWait = false;
    // Only enqueue when we're actually going to wait — otherwise stale
    // resolvers accumulate in the queue and confuse the FIFO order.
    while (!canAcquireNow(priority)) {
        if (!didWait) {
            didWait = true;
            console.warn(
                `[OnnxSlot] ${priority} acquisition BLOCKED — cap=${readMaxConcurrent()}` +
                ` inFlight(high=${inFlightHigh}, normal=${inFlightNormal})` +
                ` queued(high=${waitersHigh.length}, normal=${waitersNormal.length})`,
            );
        }
        const waiterP = new Promise<void>(resolve => queue.push(resolve));
        await waiterP;
    }
    if (priority === 'high') inFlightHigh++;
    else inFlightNormal++;
    if (didWait) {
        console.warn(`[OnnxSlot] ${priority} acquisition granted after ${Date.now() - waitStartedAt}ms`);
    }

    let released = false;
    return () => {
        if (released) return;
        released = true;
        if (priority === 'high') inFlightHigh--;
        else inFlightNormal--;
        if (didWait || waitersHigh.length > 0 || waitersNormal.length > 0) {
            console.log(
                `[OnnxSlot] ${priority} released — inFlight(high=${inFlightHigh}, normal=${inFlightNormal})` +
                ` queued(high=${waitersHigh.length}, normal=${waitersNormal.length})`,
            );
        }
        // Wake the next eligible waiter. Try high first, then normal — keeps
        // Whisper latency-critical even when embeddings are queued.
        const nextHigh = waitersHigh.shift();
        if (nextHigh) nextHigh();
        else {
            const nextNormal = waitersNormal.shift();
            if (nextNormal) nextNormal();
        }
    };
}

// ── Available-memory measurement ───────────────────────────────────────────
//
// CRITICAL: `os.freemem()` is the WRONG metric for "can I afford to load a
// model right now". On macOS it returns ONLY the truly-free page list, which
// the kernel deliberately keeps near-zero — idle RAM is used as file cache
// (inactive/speculative pages) and reclaimed instantly on demand. On a healthy
// 16-48GB Mac `os.freemem()` routinely reads 100-400MB, so a 2GB floor tested
// against it refuses EVERY local ONNX session (embedder, reranker, intent
// classifier, Whisper) essentially always — even with tens of GB reclaimable.
// This silently killed on-device embeddings/RAG for keyless users (the model
// is installed and preflight-verified, but the gate wrongly reports OOM).
//
// The right metric is AVAILABLE memory (free + reclaimable), which is what
// Activity Monitor / `top` mean by "available":
//   - macOS:  vm_stat → (free + inactive + speculative) * page_size
//   - Linux:  /proc/meminfo → MemAvailable
//   - other:  fall back to os.freemem() (best effort; Windows os.freemem() is
//             already closer to "available" than macOS's).
//
// Measurement is cached briefly (the value only needs to gate a burst of model
// loads at ingest/boot; spawning `vm_stat` per chunk would be wasteful).

const AVAIL_MEM_CACHE_TTL_MS = 1000;
let availMemCache: { gb: number; at: number } | null = null;

/** macOS: parse `vm_stat` into available GB (free + inactive + speculative). */
function readMacAvailableGB(): number | null {
    // execFileSync (no shell) — args are fixed literals, no injection surface.
    const out = execFileSync('vm_stat', [], { encoding: 'utf8', timeout: 1000 });
    // "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
    const pageSize = Number.parseInt(out.match(/page size of (\d+) bytes/)?.[1] || '4096', 10);
    const pages = (label: string): number => {
        const m = out.match(new RegExp(`${label}:\\s+(\\d+)\\.`));
        return m ? Number.parseInt(m[1], 10) : 0;
    };
    const free = pages('Pages free');
    const inactive = pages('Pages inactive');
    const speculative = pages('Pages speculative');
    if (!Number.isFinite(pageSize) || pageSize <= 0) return null;
    const bytes = (free + inactive + speculative) * pageSize;
    return bytes / 1024 ** 3;
}

/** Linux: read MemAvailable (kB) from /proc/meminfo. */
function readLinuxAvailableGB(): number | null {
    const meminfo = fs.readFileSync('/proc/meminfo', 'utf8');
    const kb = Number.parseInt(meminfo.match(/^MemAvailable:\s+(\d+)\s+kB/m)?.[1] || '', 10);
    if (!Number.isFinite(kb)) return null;
    return (kb * 1024) / 1024 ** 3;
}

/**
 * Best-effort AVAILABLE (not merely free) system memory in GB. Falls back to
 * `os.freemem()` when the platform-specific probe is unavailable or throws.
 * Cached for AVAIL_MEM_CACHE_TTL_MS to avoid spawning vm_stat per model load.
 *
 * Override for tests / incident tuning with NATIVELY_ONNX_AVAILABLE_MEM_GB
 * (a fixed value forces the gate deterministically).
 */
export function getAvailableMemoryGB(): number {
    const override = process.env.NATIVELY_ONNX_AVAILABLE_MEM_GB;
    if (override) {
        const n = Number.parseFloat(override);
        if (Number.isFinite(n) && n >= 0) return n;
    }

    const now = Date.now();
    if (availMemCache && now - availMemCache.at < AVAIL_MEM_CACHE_TTL_MS) {
        return availMemCache.gb;
    }

    let gb: number | null = null;
    try {
        if (process.platform === 'darwin') gb = readMacAvailableGB();
        else if (process.platform === 'linux') gb = readLinuxAvailableGB();
    } catch {
        gb = null;
    }
    // Fallback: os.freemem(). On Windows this is already reasonable; on
    // macOS/Linux it only lands here if the probe failed, and it's a
    // conservative (low) estimate — the gate fails toward refusing, which is
    // the pre-existing behavior, so we never regress.
    if (gb == null || !Number.isFinite(gb)) {
        gb = os.freemem() / 1024 ** 3;
    }

    availMemCache = { gb, at: now };
    return gb;
}

/**
 * Available-memory floor for admitting a new ONNX session. Returns true if the
 * system has at least `NATIVELY_ONNX_MIN_FREE_GB` (default 2.0 GB) of
 * AVAILABLE memory (free + OS-reclaimable cache), NOT merely `os.freemem()`.
 * See getAvailableMemoryGB() for why the distinction is load-bearing on macOS.
 *
 * Fails OPEN (returns true) if the measurement itself throws — refusing on
 * a measurement failure would block the app for no real reason.
 */
export function hasEnoughMemoryForOnnxSession(): boolean {
    try {
        return getAvailableMemoryGB() >= readMinFreeGB();
    } catch {
        return true;
    }
}

/** Returns the current free-memory floor in GB (live, env-aware). */
export function getMinFreeGBForOnnxSession(): number {
    return readMinFreeGB();
}

/** Returns the current max-concurrent cap (live, env-aware). */
export function getMaxConcurrentOnnxSessions(): number {
    return readMaxConcurrent();
}

/**
 * Test-only: reset the gate state so a test can re-exercise concurrent
 * acquisition from scratch. Not exported in the main barrel — only for the
 * test suite.
 */
export function __resetOnnxGateForTests(): void {
    inFlightNormal = 0;
    inFlightHigh = 0;
    waitersNormal.length = 0;
    waitersHigh.length = 0;
}
