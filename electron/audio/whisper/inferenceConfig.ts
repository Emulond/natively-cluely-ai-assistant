import type { WorkerInitMessage } from './types';

/**
 * Resolves the optimal ONNX Runtime execution providers and per-module
 * quantization (dtype) strategy for the current platform at runtime.
 *
 * Per-module dtype is the documented Whisper-safe configuration: keep the
 * encoder at fp32 (Whisper's encoder is extremely sensitive to quantization
 * — known to degrade WER several percentage points when run at int8) while
 * quantizing the decoder to q8 (decoder is token-level, much more robust to
 * quantization and dominates inference time, so the speedup is large).
 *
 * Apple Silicon (CoreML) used to take a uniform fp32 path because the ORT
 * CoreML EP had limited operator coverage for pre-quantized ONNX ops. As of
 * the 2026-07 audit the default is now the same per-module q8/fp32 map as
 * every other platform — modern CoreML handles the fp32 encoder fine and
 * the q8 decoder is the dominant speed win. A user can opt BACK to fp32
 * via the `whisperAppleSiliconDtype` setting (see resolveAppleSiliconDtype
 * below) if WER regresses on their hardware.
 */
export interface InferenceConfig {
    executionProviders: string[];
    // String → single dtype for all ONNX files (e.g. 'fp32', 'q8', 'q4').
    // Record  → per-file dtype keyed by ONNX basename without suffix:
    //           'encoder_model', 'decoder_model_merged',
    //           'decoder_model', 'decoder_with_past_model'.
    dtype: string | Record<string, string>;
}

/**
 * Whisper-safe per-module dtype map. Applies to Whisper, Distil-Whisper, and
 * Moonshine — all three use the same encoder/decoder ONNX file naming.
 *
 *   encoder_model            → fp32  (preserves acoustic encoder accuracy)
 *   decoder_model            → q8    (token decoder; quantizing here is the
 *   decoder_model_merged     → q8     standard speedup with negligible WER cost)
 *   decoder_with_past_model  → q8
 *
 * The Record acts as a SUPERSET — keys that don't match any of the loaded
 * model's actual ONNX files are silently ignored by the loader, so a single
 * map can serve all three model families (Whisper uses merged decoder,
 * Moonshine uses separate decoder + with_past, etc.).
 */
const WHISPER_SAFE_DTYPE: Record<string, string> = {
    encoder_model: 'fp32',
    decoder_model: 'q8',
    decoder_model_merged: 'q8',
    decoder_with_past_model: 'q8',
};

/**
 * Scale the catalog `sizeMb` (which is measured for the default mixed-q8
 * download: fp32 encoder + q8 decoders) toward the bytes the CURRENT platform
 * will actually download, so the progress-bar denominator (`expectedBytes`) is
 * directionally right per-platform instead of platform-blind.
 *
 * WHY THIS MATTERS: the bar denominator is `max(expectedBytes, observedTotal)`.
 * That self-corrects an UNDER-estimate (observed grows past it) but CANNOT
 * correct an OVER-estimate (the bar would finish at e.g. 65% then vanish). So
 * the only safe failure direction is to under-estimate. This factor is kept
 * deliberately conservative — at or below the true ratio — so the result stays
 * a lower bound on every platform and the un-correctable over-estimate case
 * can never occur. Being a bit low just means the bar advances slightly faster
 * early and the observed total takes over partway through, which is smooth.
 *
 *   - Apple Silicon resolves uniform fp32 (see resolveInferenceConfig): the q8
 *     decoders are instead downloaded at fp32, so the real download is larger
 *     than the catalog q8 figure. A factor >1 keeps expectedBytes a lower bound
 *     while starting far closer to reality. 1.6 is intentionally below the
 *     true fp32/q8 ratio (~2–3× on the decoder-heavy portion) so we never
 *     over-shoot.
 *   - Everything else already matches the catalog's mixed-q8 measurement → 1.0.
 */
function dtypeSizeFactor(dtype: string | Record<string, string>): number {
    // Uniform fp32 across all modules = the Apple Silicon / large-download path.
    if (dtype === 'fp32') return 1.6;
    // Mixed per-module map (WHISPER_SAFE_DTYPE) or any q8/q4 string: the catalog
    // figure already reflects this, so no scaling.
    return 1.0;
}

/**
 * Construct the worker `init` message for a given model. Single source of
 * truth — three callers (LocalWhisperSTT.spawnWorker, modelPreloader.preload,
 * local-whisper-start-download IPC) all use this so the message shape stays
 * consistent. The cacheDir lookup is lazy (avoids importing electron from
 * this leaf module).
 */
export function buildWorkerInitMessage(
    modelId: string,
    opts?: { forDownload?: boolean },
): WorkerInitMessage {
    // Late require — modelManager imports electron, which isn't available
    // when this module is first loaded in some contexts (test harnesses).
    const { getModelsDir, getModelSizeBytes, getModelExternalDataFormat } = require('./modelManager');
    const { executionProviders, dtype } = resolveInferenceConfig();
    // Catalog download size — progress-bar denominator from byte zero. The
    // lookup is best-effort: if it's missing (unknown id) or the call fails
    // for any reason, we send 0 and the worker falls back to summing the
    // per-file byte totals it observes during the download. The size is a
    // UX nicety for the progress bar, never required for the download itself,
    // so a failure here must NEVER prevent the worker from starting.
    const extraDtypes = opts?.forDownload ? resolveExtraDownloadDtypes(gpuDeviceIdForDownload()) : undefined;
    let expectedBytes = 0;
    try {
        // The extra GPU precision is part of the SAME download, so it has to be
        // part of the same denominator. Without this the bar reached 100% when
        // the mixed set finished and then kept fetching ~484MB more with no
        // indication — a download that looks finished but is not is worse than a
        // slow one, because the natural response is to close the app, which
        // leaves the model half-fetched and reported "not cached" at next start.
        //
        // 1.8x is deliberately BELOW the true ratio (whisper-small: 510MB mixed
        // + 484MB fp16 ≈ 1.95x). Over-estimating cannot self-correct — the bar
        // would stop at ~90% and vanish — while under-estimating just means the
        // observed total takes over partway through, which is smooth. Same
        // reasoning as dtypeSizeFactor above.
        const extraFactor = extraDtypes && extraDtypes.length > 0 ? 1.8 : 1.0;
        const n = Number(getModelSizeBytes(modelId)) * dtypeSizeFactor(dtype) * extraFactor;
        if (Number.isFinite(n) && n > 0) expectedBytes = Math.round(n);
    } catch {
        expectedBytes = 0;
    }
    // External-data flag for checkpoints whose weights live in sibling
    // `*.onnx_data` files but whose own config.json doesn't declare it (e.g.
    // Whisper Large v3 Turbo). undefined for every other model — the worker
    // then lets transformers read each model's config.json as before. Like the
    // size lookup above, never let this block worker startup.
    let useExternalDataFormat: boolean | Record<string, boolean> | undefined;
    try {
        useExternalDataFormat = getModelExternalDataFormat(modelId);
    } catch {
        useExternalDataFormat = undefined;
    }
    // Whichever DirectML adapter the startup probe proved usable, or null. The
    // probe runs in a child process precisely so that this lookup can never be
    // the thing that crashes a meeting — by the time it is read, the dangerous
    // part already happened somewhere expendable.
    let gpuDeviceId: number | null = null;
    let gpuQuantizedOk = false;
    let gpuReason = 'GPU probe has not finished yet — starting on CPU';
    try {
        const { getResolvedGpuDevice } = require('./gpuProbe');
        const probe = getResolvedGpuDevice();
        if (probe) {
            gpuDeviceId = probe.deviceId;
            gpuQuantizedOk = !!probe.quantizedOk;
            gpuReason = probe.reason;
        }
    } catch {
        gpuReason = 'GPU probe unavailable — using CPU';
    }
    return {
        type: 'init',
        modelId,
        cacheDir: getModelsDir(),
        executionProviders,
        dtype,
        expectedBytes,
        useExternalDataFormat,
        gpuDeviceId,
        gpuQuantizedOk,
        gpuReason,
        extraDtypes,
        concurrentSessions: resolveActiveWhisperChannelCount(),
    };
}

/**
 * How many Whisper sessions will actually be live at once.
 *
 * The intra-op thread budget is split across concurrent sessions so two
 * channels cannot oversubscribe the machine. Hardcoding that divisor at 2
 * meant a user who turned one channel OFF still had the survivor running on
 * half the cores — paying the cost of a session that no longer exists.
 *
 * Counts what is configured, not what happens to be running: the count is read
 * when a worker is created, and both channels start within the same moment of
 * meeting setup.
 */
export function resolveActiveWhisperChannelCount(): number {
    try {
        const { SettingsManager } = require('../../services/SettingsManager');
        const { isChannelDisabled } = require('./modelManager');
        const sm = SettingsManager.getInstance();
        if (!sm.get('localWhisperPerChannelEnabled')) return 2;
        const off = [sm.get('localWhisperModelMic'), sm.get('localWhisperModelSystem')]
            .filter((id: string | undefined) => isChannelDisabled(id)).length;
        return Math.max(1, 2 - off);
    } catch {
        // Settings unreachable (test contexts) — assume both, which is the
        // conservative direction: fewer threads each, never oversubscribed.
        return 2;
    }
}

/** The probed adapter, read on its own so expectedBytes can be sized before the
 *  main probe block below runs. Same source, no second probe. */
function gpuDeviceIdForDownload(): number | null {
    try {
        const { getResolvedGpuDevice } = require('./gpuProbe');
        return getResolvedGpuDevice()?.deviceId ?? null;
    } catch {
        return null;
    }
}

/**
 * Extra precisions worth fetching while the user is already waiting on a
 * download.
 *
 * A model's normal download is an fp32 encoder plus QUANTISED decoders. That is
 * the right set for CPU inference and the wrong set for a GPU that refuses
 * int8 — and DirectML's int8 operator coverage is narrow enough that refusing
 * is common. Discovering the gap at meeting time leaves two bad options: start
 * a several-hundred-MB download under someone who just pressed Start Live
 * Meeting, or fall back to the CPU and be slow.
 *
 * WHY fp16 AND NOT fp32. Both are published; the file listing for
 * Xenova/whisper-small settles it:
 *
 *   decoder_model_merged.onnx        615,405,212 B   (fp32)
 *   decoder_model_merged_fp16.onnx   308,615,077 B   (fp16)
 *   encoder_model.onnx               352,839,389 B   (fp32)
 *   encoder_model_fp16.onnx          176,608,338 B   (fp16)
 *
 * fp16 is half the download, half the VRAM — 484MB against 968MB, the
 * difference between comfortable and tight on a 4GB laptop card — and it is
 * DirectML's native precision rather than a tolerated one. Turing and later run
 * fp16 at twice the fp32 rate. There is no axis on which fp32 wins here.
 *
 * Machines with no usable GPU fetch nothing extra: there is no path on which it
 * would ever be read.
 */
export function resolveExtraDownloadDtypes(
    gpuDeviceId: number | null,
): Array<string | Record<string, string>> | undefined {
    if (gpuDeviceId === null || gpuDeviceId === undefined) return undefined;
    return ['fp16'];
}

/**
 * Apple Silicon dtype override — lets a user opt back to uniform fp32 if
 * the new per-module q8 default regresses WER on their hardware. Read from
 * SettingsManager (`whisperAppleSiliconDtype`); missing/unknown → the new
 * per-module default. Returns null on SettingsManager unavailable (test
 * contexts) so the resolver falls through to its own default.
 */
function resolveAppleSiliconDtype(): string | Record<string, string> | null {
    try {
        // Lazy require: SettingsManager touches electron's app, which isn't
        // available in unit-test contexts. Any throw here means "no override".
        const { SettingsManager } = require('../../services/SettingsManager');
        const raw = SettingsManager.getInstance().get('whisperAppleSiliconDtype');
        if (raw === 'fp32' || raw === 'q8' || raw === 'q4' || raw === 'int8') {
            return 'fp32';
        }
        if (raw === 'mixed') {
            return WHISPER_SAFE_DTYPE;
        }
        return null; // unknown / not set → caller uses its default
    } catch {
        return null;
    }
}

/**
 * Cross-platform dtype override via NATIVELY_WHISPER_DTYPE.
 *
 * The default keeps the encoder at fp32 for accuracy, but the encoder is the
 * dominant cost of a Whisper pass: it runs over a full 30-SECOND mel window
 * every time, however briefly the user actually spoke. On CPU that is what
 * makes short utterances expensive — a measured 6959ms for 0.81s of speech on
 * whisper-small (realtimeFactor 8.59x).
 *
 * Quantizing the encoder is the standard trade for realtime use, so this exists
 * to make that reachable without a rebuild:
 *
 *   NATIVELY_WHISPER_DTYPE=q8    — quantize everything, encoder included
 *   NATIVELY_WHISPER_DTYPE=fp32  — uniform fp32 (maximum accuracy)
 *   NATIVELY_WHISPER_DTYPE=mixed — the shipped default (fp32 encoder, q8 decoders)
 *
 * Deliberately NOT the default: it costs word accuracy, and that is a product
 * decision rather than one to make silently on a user's behalf.
 */
function resolveEnvDtype(): string | Record<string, string> | null {
    const raw = (process.env.NATIVELY_WHISPER_DTYPE ?? '').trim().toLowerCase();
    if (!raw) return null;
    if (raw === 'mixed') return WHISPER_SAFE_DTYPE;
    if (raw === 'q8' || raw === 'fp32' || raw === 'q4' || raw === 'int8' || raw === 'fp16') return raw;
    console.warn(`[inferenceConfig] NATIVELY_WHISPER_DTYPE="${raw}" not recognised — using the platform default`);
    return null;
}

export function resolveInferenceConfig(): InferenceConfig {
    const { platform, arch } = process;
    const envDtype = resolveEnvDtype();

    if (platform === 'darwin' && arch === 'arm64') {
        // Apple Silicon — CoreML uses Metal GPU + ANE. Default changed in
        // 2026-07 from uniform fp32 → mixed per-module (fp32 encoder + q8
        // decoders), matching every other platform. The q8 decoder is the
        // dominant speed win and modern CoreML handles the mixed-precision
        // graph cleanly. A user can override back to fp32 via the
        // `whisperAppleSiliconDtype` setting if their WER regresses.
        const override = resolveAppleSiliconDtype();
        return {
            executionProviders: ['coreml', 'cpu'],
            dtype: envDtype ?? override ?? WHISPER_SAFE_DTYPE,
        };
    }

    if (platform === 'win32') {
        // Windows — DirectML over NVIDIA / AMD / Intel GPUs. Per-module dtype
        // gives best accuracy/speed tradeoff for the larger Whisper/Distil
        // checkpoints; DirectML handles mixed precision via session options.
        return { executionProviders: ['dml', 'cpu'], dtype: envDtype ?? WHISPER_SAFE_DTYPE };
    }

    // Intel Mac, Linux, unknown — CPU. Per-module gives a real speedup on
    // decoder-heavy inference without sacrificing encoder accuracy.
    return { executionProviders: ['cpu'], dtype: envDtype ?? WHISPER_SAFE_DTYPE };
}
