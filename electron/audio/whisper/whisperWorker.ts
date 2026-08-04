/**
 * Node.js Worker Thread for ASR inference via @huggingface/transformers v3+.
 *
 * Supports two model families:
 *   - Whisper (and Distil-Whisper): batch-architected, 30s windows, slow but
 *     widely supported and multilingual.
 *   - Moonshine: streaming-architected with encoder caching + decoder state
 *     reuse, ~100× lower latency than Whisper Large v3 at comparable WER.
 *     English-only. Models load in 26–60MB quantized.
 *
 * @huggingface/transformers is ESM-only. The electron tsconfig compiles to
 * CommonJS, which means TypeScript rewrites `import()` to `require()`.
 * We bypass this by loading the package through `new Function(...)` so
 * the compiler never sees the import expression and Node.js handles it
 * natively as a true dynamic ESM import at runtime.
 */
import { parentPort } from 'worker_threads';
import { WhisperProgressAggregator } from './whisperProgressAggregator';
import { getBoundedOnnxSessionOptions } from '../../utils/onnxThreadConfig';
import { RECOGNITION_LANGUAGES } from '../../config/languages';

// The host sends the shared RECOGNITION_LANGUAGES key ('russian', 'english-us',
// 'auto') — the same value every other STT provider receives — NOT a BCP-47 tag.
// This used to be a hand-rolled BCP-47-keyed table, so every selection missed it
// and fell through to null: picking Russian transcribed Russian speech as English.
// Resolve through the shared config instead, reading iso639 (Whisper's language
// tokens are ISO-639-1) like OpenAI/Groq/Deepgram/ElevenLabs/Soniox already do.
// That also covers all ~35 configured languages rather than a stale subset of 15.
//
// 'auto' (and an unknown key) resolves to null, which leaves the language
// unforced so a multilingual checkpoint runs its own detection.
function resolveWhisperLanguage(key: string): string | null {
  if (!key || key === 'auto') return null;
  return RECOGNITION_LANGUAGES[key]?.iso639 ?? null;
}

let pipe: any = null;
let loadedModelId = '';

// Tokenized prompt cache — populated by `setPrompt` messages, reused by
// every subsequent transcribe. Cleared on model swap.
//
// The transcribe message handler must remain serial w.r.t. setPrompt so we
// don't read a half-updated cache; the host-side caller (LocalWhisperSTT)
// posts setPrompt via the same MessagePort which Node guarantees orders
// strictly with transcribe messages. As long as no two transcribe messages
// are in flight concurrently (the streamingTaskInFlight guard ensures this),
// the cache is consistent.
let cachedPromptText = '';
let cachedPromptIds: number[] | null = null;

// Moonshine doesn't have Whisper's prompt_ids mechanism. Detect by model id
// so we silently skip the prompt parameter for Moonshine variants.
const isMoonshineModel = (id: string) => /\/moonshine-/i.test(id);

const PROMPT_TOKEN_CAP = 224; // Whisper's prompt window per generation_whisper.js

async function updatePromptCache(promptText: string): Promise<void> {
  const trimmed = (promptText ?? '').trim();
  if (!trimmed) {
    cachedPromptText = '';
    cachedPromptIds = null;
    return;
  }
  if (trimmed === cachedPromptText && cachedPromptIds !== null) return;
  if (!pipe?.tokenizer) return; // model not yet loaded
  if (isMoonshineModel(loadedModelId)) {
    // Skip tokenization entirely for Moonshine — no prompt mechanism.
    cachedPromptText = trimmed;
    cachedPromptIds = null;
    return;
  }
  try {
    // add_special_tokens=false: Whisper inserts <|startofprev|> itself.
    const encoded = await pipe.tokenizer(trimmed, { add_special_tokens: false });
    const raw = encoded?.input_ids?.tolist?.()?.[0] ?? [];
    // Truncate from the END (keep first 224). Session-static biasing prompts
    // typically front-load the most important vocabulary (attendee names,
    // company/project names, glossary terms), so dropping the tail of less
    // important tokens preserves the user's priority order.
    cachedPromptIds = raw.slice(0, PROMPT_TOKEN_CAP).map((n: bigint | number) => {
      const v = Number(n);
      // Whisper vocab is ~50k tokens — well under 2^53 — but if a future
      // model ships sentinel ids with high bits set, fail loud rather than
      // silently bias on a precision-lost token id.
      if (!Number.isSafeInteger(v)) {
        throw new Error(`Token id ${n} exceeds Number.MAX_SAFE_INTEGER — cannot use as prompt_id`);
      }
      return v;
    });
    cachedPromptText = trimmed;
    if (cachedPromptIds.length === 0) {
      console.debug('[WhisperWorker] Prompt tokenized to 0 ids — biasing disabled');
    }
  } catch (e: any) {
    console.warn('[WhisperWorker] Prompt tokenization failed:', e.message);
    cachedPromptText = '';
    cachedPromptIds = null;
  }
}

// English-only checkpoints have NO multilingual decoder, and transformers.js
// refuses BOTH `language` and `task` for them:
//
//   Error: Cannot specify `task` or `language` for an English-only model.
//     at _retrieve_init_tokens (transformers.node.mjs)
//
// It throws before a single audio frame is decoded, so every transcription
// fails and the channel produces no text at all while the capture side looks
// perfectly healthy. Observed on a packaged Windows build, three for three:
//
//   transcribe START  task=t1 lang=english model=Xenova/whisper-tiny.en
//   transcribe FAILED task=t1: Cannot specify `task` or `language` ...
//
// So for these models we must send NEITHER option — not `language: 'english'`,
// which is what the previous code did. They transcribe English unconditionally;
// omitting both is exactly the behaviour we wanted anyway.
const ENGLISH_ONLY_MODELS = new Set([
  // Moonshine — English-only by design
  'onnx-community/moonshine-tiny-ONNX',
  'onnx-community/moonshine-base-ONNX',
  // Distil-Whisper — English-only checkpoints
  'distil-whisper/distil-small.en',
  'distil-whisper/distil-medium.en',
  'distil-whisper/distil-large-v2',
  'distil-whisper/distil-large-v3',
  // Whisper .en variants
  'Xenova/whisper-tiny.en',
  'Xenova/whisper-base.en',
  'Xenova/whisper-small.en',
  'Xenova/whisper-medium.en',
]);

// Checkpoints that proved English-only at runtime. The static list above cannot
// know about a checkpoint added later, and being wrong about one costs the whole
// session's transcript — so a refusal is recorded here and the next segment goes
// out clean. Exactly one segment pays for the discovery instead of all of them.
const englishOnlyAtRuntime = new Set<string>();

// transformers.js decides this from `generation_config.is_multilingual`, so read
// the same flag off the loaded model rather than guessing from the id.
function isEnglishOnlyModel(id: string): boolean {
  if (ENGLISH_ONLY_MODELS.has(id) || englishOnlyAtRuntime.has(id)) return true;
  return pipe?.model?.generation_config?.is_multilingual === false;
}

const isEnglishOnlyRefusal = (e: any): boolean =>
  /English-only model/i.test(String(e?.message ?? e ?? ''));

if (!parentPort) throw new Error('whisperWorker must be run as a Worker thread');

// Forward this worker's console output to the host so it lands in
// natively_debug.log. A worker_thread's stdout is NOT piped into the main
// process log, so everything this file reports — model load progress, ONNX
// provider selection, decode errors — was invisible in user logs. That blind
// spot is precisely where "audio dispatched, no transcript" failures live:
// the host can see that it POSTED a transcribe and got nothing back, but not
// why. Best-effort and non-fatal: a failed post must never break inference.
for (const level of ['log', 'warn', 'error'] as const) {
  const original = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    original(...args);
    try {
      const message = args
        .map((a) => {
          if (typeof a === 'string') return a;
          if (a instanceof Error) return a.stack ?? a.message;
          try { return JSON.stringify(a); } catch { return String(a); }
        })
        .join(' ');
      parentPort!.postMessage({ type: 'log', level, message });
    } catch { /* forwarding is best-effort — never break the worker over a log */ }
  };
}

// Loads @huggingface/transformers via a real dynamic import() at runtime.
// Using new Function prevents TypeScript from rewriting import() → require()
// in the CommonJS output, which would fail because the package is ESM-only.
async function loadTransformers(): Promise<{ pipeline: any; env: any }> {
  return (new Function('return import("@huggingface/transformers")')()) as any;
}

parentPort.on('message', async (msg: any) => {
  if (msg.type === 'init') {
    // Validate required fields BEFORE entering the try/catch so the error
    // surfaces as a structured `error` postMessage rather than an unhandled
    // worker throw (which would leave the host's workerReady stuck false).
    if (msg.dtype === undefined || msg.dtype === null) {
      parentPort!.postMessage({
        type: 'error',
        message: 'init.dtype is required (use resolveInferenceConfig().dtype)',
      });
      return;
    }
    try {
      const { pipeline, env } = await loadTransformers();

      env.cacheDir = msg.cacheDir;
      env.allowRemoteModels = true;

      // Apply hardware-specific execution providers (CoreML, DirectML, CUDA, CPU)
      const providers: string[] = msg.executionProviders ?? ['cpu'];
      if (env.backends?.onnx) {
        env.backends.onnx.executionProviders = providers;
      }
      // Per-module dtype: required. @huggingface/transformers v3 no longer
      // honors the v2 `quantized: true` flag — must use `dtype` explicitly.
      const dtype: string | Record<string, string> = msg.dtype;
      // Sort entries for deterministic log output across runs.
      const dtypeDesc = typeof dtype === 'string'
        ? dtype
        : 'mixed:' + Object.entries(dtype).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => `${k}=${v}`).join(',');
      const sessionOptions = getBoundedOnnxSessionOptions();

      console.log(`[WhisperWorker] Loading ${msg.modelId} | providers=${providers.join(',')} | dtype=${dtypeDesc}`);

      // DIAGNOSTICS (2026-06-13): the model files load fine in isolation (raw ORT +
      // transformers, both in system node), yet the live worker can fail with
      // "Protobuf parsing failed". Log the exact runtime view so the failing GUI run
      // prints precisely WHY — cacheDir, resolved file paths + sizes, ORT backend, and
      // the ORT version transformers actually bound. Cheap, init-only (not per-token).
      try {
        const _fs = require('fs');
        const _path = require('path');
        const _orgName = String(msg.modelId).split('/');
        const _modelDir = _path.join(String(msg.cacheDir), _orgName[0] || '', _orgName[1] || '', 'onnx');
        const _encName = typeof dtype === 'string' && dtype !== 'fp32' ? `encoder_model_${dtype}.onnx` : 'encoder_model.onnx';
        const _decName = typeof dtype === 'string' && dtype !== 'fp32' ? `decoder_model_merged_${dtype}.onnx` : 'decoder_model_merged.onnx';
        const _stat = (p: string) => { try { return _fs.statSync(p).size; } catch { return -1; } };
        let _ortVer = 'unknown';
        try { _ortVer = require('onnxruntime-node/package.json').version; } catch { /* bundled? */ }
        console.log('[WhisperWorker][diag]', JSON.stringify({
          cacheDir: String(msg.cacheDir),
          modelDir: _modelDir,
          modelDirExists: _fs.existsSync(_modelDir),
          encoderFile: _encName, encoderBytes: _stat(_path.join(_modelDir, _encName)),
          decoderFile: _decName, decoderBytes: _stat(_path.join(_modelDir, _decName)),
          providers, dtype: dtypeDesc,
          sessionOptions,
          ortNodeVersion: _ortVer,
          ortBackend: (env.backends?.onnx ? Object.keys(env.backends.onnx) : []),
          execEnv: { execPath: process.execPath, nodeVer: process.version, modules: process.versions.modules, electron: process.versions.electron || 'n/a' },
        }));
      } catch (diagErr: any) {
        console.log('[WhisperWorker][diag] diagnostics failed (non-fatal):', diagErr?.message);
      }

      // HF Transformers fires progress_callback per *file* (encoder, decoder,
      // tokenizer, config…). The raw `data.progress` is per-file 0..100, which
      // makes a model-level bar bounce around (3 → 2 → 100 → 5 → …) as files
      // start, complete, and new ones enter the stream. The byte-weighted
      // aggregation that turns those per-file events into a smooth model-level
      // percentage lives in whisperProgressAggregator.ts (pure + unit-tested);
      // see that file for the full rationale on why count-averaging produced
      // the old "jumps to ~80% then stalls" bug.
      //
      // expectedBytes = catalog download size, the denominator from byte zero.
      // 0 when unknown / lookup failed → the aggregator falls back to observed
      // file totals. The constructor sanitizes any non-finite/negative value.
      const aggregator = new WhisperProgressAggregator(Number(msg.expectedBytes));
      // External-data format: forwarded only when the catalog declares it (for
      // checkpoints whose config.json omits it, e.g. Whisper Large v3 Turbo).
      // When undefined, transformers falls back to the model's own config —
      // preserving prior behaviour for every self-declaring model. Without this
      // the sibling `*.onnx_data` weight file is never fetched and ORT aborts:
      // "filesystem error: in file_size: ... encoder_model.onnx_data".
      const useExternalDataFormat: boolean | Record<string, boolean> | undefined =
        msg.useExternalDataFormat;
      const loadT0 = Date.now();

      // resolveInferenceConfig() asks for DirectML on Windows and CoreML on
      // Apple Silicon, but that list was only ever assigned to
      // env.backends.onnx.executionProviders — which transformers.js does not
      // read. Execution provider selection goes through the `device` option on
      // pipeline(). Without it every model silently ran on CPU, so the GPU path
      // this app has always intended to use was inert: measured 6959ms to
      // transcribe 0.81s of speech (realtimeFactor 8.59x) on whisper-small.
      //
      // Only pass devices transformers.js actually knows (DEVICE_TYPES). Notably
      // 'coreml' is NOT one of them, so macOS keeps the previous behaviour of
      // letting the library choose rather than being handed an invalid value.
      // ── GPU acceleration ────────────────────────────────────────────────
      //
      // onnxruntime-node ships CPU + DirectML on win32; CUDA is Linux-only in
      // that binding, so DirectML is the only GPU path on Windows regardless of
      // GPU vendor. DirectML runs over DX12 and drives NVIDIA/AMD/Intel alike.
      //
      // WHICH ADAPTER MATTERS. On a laptop, DirectML's default adapter 0 is
      // typically the integrated GPU, which shares system RAM — handing it a
      // 352MB fp32 encoder is how the earlier device='dml' attempt took the
      // whole app down at startup (native abort during init: no JS throw, so a
      // try/catch cannot rescue it). A discrete GPU with its own VRAM is a very
      // different proposition. ORT's DmlExecutionProviderOption accepts
      // `deviceId`, so the adapter is selectable — set NATIVELY_ONNX_DEVICE_ID.
      //
      // HOW THIS REACHES ORT. Passing transformers.js a `device` string gives no
      // way to specify deviceId. But it assigns its own provider list with
      //     session_options.executionProviders ??= executionProviders;
      // — `??=`, so a list we supply is preserved. We therefore build the EP
      // list ourselves and pass it through session_options, with 'cpu' appended
      // so ORT falls back per-operator for anything DirectML cannot run.
      //
      // CRASH SAFETY. A GPU init that aborts the process would otherwise make
      // the app unopenable forever. Before attempting one we drop a sentinel
      // file and clear it once the model is loaded; finding a stale sentinel at
      // startup means the previous attempt never survived, so we stay on CPU and
      // say so. Worst case is a single crash, then permanent automatic fallback.
      // This mirrors the writeLoadSentinel/consumePoisonedOnnxLoad pattern used
      // for model loads elsewhere in this codebase.
      const GPU_DEVICES = new Set(['dml', 'cuda', 'webgpu']);
      const requestedDevice = (process.env.NATIVELY_ONNX_DEVICE ?? '').trim().toLowerCase();
      const deviceIdRaw = Number.parseInt(process.env.NATIVELY_ONNX_DEVICE_ID ?? '', 10);
      const deviceId = Number.isInteger(deviceIdRaw) && deviceIdRaw >= 0 ? deviceIdRaw : undefined;

      const _gpuFs = require('fs');
      const _gpuPath = require('path');
      const gpuSentinelPath = _gpuPath.join(msg.cacheDir ?? '.', '.gpu-init-sentinel.json');
      const readGpuSentinel = (): any | null => {
        try {
          if (!_gpuFs.existsSync(gpuSentinelPath)) return null;
          return JSON.parse(_gpuFs.readFileSync(gpuSentinelPath, 'utf8'));
        } catch { return null; }
      };
      const writeGpuSentinel = (info: unknown): void => {
        try { _gpuFs.writeFileSync(gpuSentinelPath, JSON.stringify(info)); } catch { /* best-effort */ }
      };
      const clearGpuSentinel = (): void => {
        try { if (_gpuFs.existsSync(gpuSentinelPath)) _gpuFs.unlinkSync(gpuSentinelPath); } catch { /* best-effort */ }
      };

      // AUTOMATIC GPU LADDER. Requiring a user to set environment variables to
      // get their GPU used is not a fix, so this now configures itself.
      //
      // Adapter order cannot be queried from Node, but on a laptop the discrete
      // GPU is conventionally adapter 1 and the integrated one adapter 0 — and
      // the integrated GPU is what crashed when DirectML defaulted to adapter 0.
      // So the ladder tries the discrete GPU first, then the integrated one,
      // then CPU, recording each attempt so a failure is never repeated:
      //
      //   attempt 1 → dml deviceId=1   (discrete GPU, e.g. a laptop NVIDIA)
      //   attempt 2 → dml deviceId=0   (integrated GPU)
      //   attempt 3 → CPU              (always works, never retried past here)
      //
      // The sentinel is written BEFORE each attempt and cleared on success, so a
      // native abort — which no JS catch can intercept — is still recorded. The
      // next launch reads it and moves down the ladder. A machine that hates
      // DirectML costs at most two startups before settling permanently on CPU.
      //
      // NATIVELY_ONNX_DEVICE=cpu opts out entirely; NATIVELY_ONNX_DEVICE=dml
      // plus NATIVELY_ONNX_DEVICE_ID=N pins a specific adapter.
      const isWindows = process.platform === 'win32';
      let chosenDevice: string | undefined;
      let chosenDeviceId: number | undefined;

      if (requestedDevice === 'cpu') {
        console.log('[WhisperWorker] GPU disabled by NATIVELY_ONNX_DEVICE=cpu');
      } else if (GPU_DEVICES.has(requestedDevice)) {
        chosenDevice = requestedDevice;
        chosenDeviceId = deviceId;
      } else if (isWindows && (process.env.NATIVELY_ONNX_GPU_LADDER ?? '') === '1') {
        // OFF BY DEFAULT. The automatic ladder crashed the app on launch, and
        // the sentinel could not save it: a native DirectML abort can kill the
        // process before the sentinel write reaches disk, so the next start
        // reads no record, tries the same adapter, and crashes again — an
        // unopenable app rather than the one-crash-then-fallback this was
        // supposed to guarantee.
        //
        // An app that will not open is worse than one that transcribes slowly,
        // so GPU stays opt-in until the attempt itself can be made survivable
        // (a probe in a throwaway child process, whose death is observable
        // instead of fatal).
        chosenDevice = 'dml';
        chosenDeviceId = deviceId ?? 1;
      }

      if (chosenDevice) {
        const poisoned = readGpuSentinel();
        if (poisoned) {
          const failedId = typeof poisoned.deviceId === 'number' ? poisoned.deviceId : undefined;
          if (deviceId !== undefined) {
            // An explicit pin already failed — respect it and stop.
            console.warn(`[WhisperWorker] SKIPPING GPU: pinned device ${chosenDevice}:${deviceId} previously failed to initialise. Using CPU.`);
            chosenDevice = undefined;
          } else if (failedId === 1) {
            console.warn('[WhisperWorker] GPU adapter 1 previously failed to initialise — trying adapter 0 (integrated).');
            chosenDeviceId = 0;
          } else {
            console.warn(`[WhisperWorker] GPU previously failed to initialise (${JSON.stringify(poisoned)}) — using CPU. Delete ${gpuSentinelPath} to retry.`);
            chosenDevice = undefined;
          }
        }
      }

      // Our own EP list wins over transformers.js's, per the ??= above.
      const gpuExecutionProviders = chosenDevice
        ? [chosenDeviceId !== undefined ? { name: chosenDevice, deviceId: chosenDeviceId } : { name: chosenDevice }, 'cpu']
        : undefined;
      const preferredDevice = chosenDevice;
      const deviceIdInUse = chosenDeviceId;

      const buildPipeline = (device?: string) => pipeline('automatic-speech-recognition', msg.modelId, {
        dtype,
        session_options: device && gpuExecutionProviders
          ? { ...sessionOptions, executionProviders: gpuExecutionProviders }
          : sessionOptions,
        ...(device ? { device } : {}),
        ...(useExternalDataFormat !== undefined
          ? { use_external_data_format: useExternalDataFormat }
          : {}),
        progress_callback: (data: any) => {
          const { pct } = aggregator.update(data);
          if (pct === null) return;
          parentPort!.postMessage({
            type: 'progress',
            modelId: msg.modelId,
            progress: pct,
          });
        },
      });

      try {
        if (preferredDevice) {
          writeGpuSentinel({ device: preferredDevice, deviceId: deviceIdInUse ?? 'default', modelId: msg.modelId, at: Date.now() });
          console.log(
            `[WhisperWorker] building pipeline with device=${preferredDevice} deviceId=${deviceIdInUse ?? '(ORT default)'}` +
            ` providers=${JSON.stringify(gpuExecutionProviders)}`,
          );
        } else {
          console.log('[WhisperWorker] building pipeline on CPU (library default device)');
        }
        pipe = await buildPipeline(preferredDevice);
        // Survived initialisation — this configuration is safe to retry.
        clearGpuSentinel();
      } catch (deviceErr: any) {
        // A GPU provider can fail for reasons entirely outside our control —
        // no compatible adapter, a driver that rejects the graph, a DirectML
        // version mismatch. Falling back to CPU keeps transcription working
        // (slowly) instead of leaving the channel with no worker at all, which
        // is silent from the user's side.
        if (!preferredDevice || preferredDevice === 'cpu') { clearGpuSentinel(); throw deviceErr; }
        // A clean JS throw (as opposed to a native abort) is recoverable — but
        // keep the sentinel so the next launch does not retry a device that
        // already failed here.
        console.warn(
          `[WhisperWorker] device=${preferredDevice} failed to initialise (${deviceErr?.message ?? deviceErr}) — falling back to CPU`,
        );
        pipe = await buildPipeline(undefined);
      }
      loadedModelId = msg.modelId;

      // What actually got built. pipeline() is called WITHOUT a `device`
      // option, so transformers.js picks its own default for Node rather than
      // honouring env.backends.onnx.executionProviders — meaning the
      // "providers=dml,cpu" line logged above may bear no relation to the
      // execution provider really in use. Introspect the constructed session
      // instead of trusting what we requested. Entirely best-effort: the shape
      // of these internals is not part of the transformers.js public API.
      console.log(`[WhisperWorker] model READY in ${Date.now() - loadT0}ms for ${msg.modelId}`);
      try {
        const model: any = (pipe as any)?.model;
        const sessions = model?.sessions ?? {};
        const sessionNames = Object.keys(sessions);
        const epsBySession: Record<string, unknown> = {};
        for (const name of sessionNames) {
          const s: any = sessions[name];
          epsBySession[name] =
            s?.handler?.executionProviders ??
            s?.executionProviders ??
            s?.handler?._executionProviders ??
            'unknown';
        }
        console.log('[WhisperWorker][diag:session] ' + JSON.stringify({
          modelClass: model?.constructor?.name ?? 'unknown',
          deviceUsed: preferredDevice ? `${preferredDevice}:${deviceIdInUse ?? 'default'}` : 'cpu (library default)',
          sessions: sessionNames,
          executionProviders: epsBySession,
          requestedProviders: providers,
          sessionOptions,
        }));
      } catch (introspectErr: any) {
        console.log(`[WhisperWorker][diag:session] introspection failed (non-fatal): ${introspectErr?.message}`);
      }

      // New model = stale prompt cache (different tokenizer vocab)
      cachedPromptText = '';
      cachedPromptIds = null;

      parentPort!.postMessage({ type: 'ready' });
    } catch (e: any) {
      // Full failure dump (2026-06-13 diag): the error message alone ("Protobuf
      // parsing failed") doesn't say WHICH file or WHY. Log the full error, stack,
      // and any ORT-specific cause so the failing GUI run is self-diagnosing.
      try {
        console.error('[WhisperWorker][diag] MODEL LOAD FAILED:', {
          modelId: msg.modelId,
          message: e?.message,
          name: e?.name,
          code: e?.code,
          cause: e?.cause ? String(e.cause).slice(0, 300) : undefined,
          stackHead: String(e?.stack || '').split('\n').slice(0, 5).join(' | '),
        });
      } catch { /* noop */ }
      parentPort!.postMessage({
        type: 'error',
        message: `Failed to load model: ${e.message}`,
      });
    }
  } else if (msg.type === 'setPrompt') {
    await updatePromptCache(msg.prompt);
  } else if (msg.type === 'transcribe') {
    if (!pipe) {
      parentPort!.postMessage({ type: 'error', message: 'Model not loaded' });
      return;
    }
    try {
      let language: string | null = resolveWhisperLanguage(msg.language);
      const streaming: boolean = !!msg.streaming;

      // English-only checkpoints reject `language` AND `task` outright — see
      // ENGLISH_ONLY_MODELS. Drop the language here and the task below.
      const englishOnly = isEnglishOnlyModel(loadedModelId);
      if (englishOnly) language = null;

      // Streaming partial passes use deterministic settings so consecutive
      // overlapping windows are stable enough for LocalAgreement-2 to
      // converge on a committed prefix. Final passes also disable
      // condition_on_previous_text + add Whisper's standard fallback
      // thresholds to suppress repetition loops on long segments.
      const opts: any = streaming
        ? {
            sampling_rate: 16000,
            task: 'transcribe',
            temperature: 0,
            no_speech_threshold: 0.6,
            // Whisper's anti-loop check — drops outputs whose token gzip
            // ratio exceeds 2.4 (typical of "thank you. thank you. thank
            // you..." hallucinations on near-silent windows). Final pass
            // uses the same threshold; streaming should match for
            // consistency in what reaches the user.
            compression_ratio_threshold: 2.4,
            condition_on_previous_text: false,
            return_timestamps: false,
          }
        : {
            sampling_rate: 16000,
            task: 'transcribe',
            condition_on_previous_text: false,
            compression_ratio_threshold: 2.4,
            logprob_threshold: -1.0,
            no_speech_threshold: 0.6,
          };
      if (language) opts.language = language;
      // `task` is refused alongside `language` on English-only checkpoints, and
      // 'transcribe' is what they do anyway — there is nothing to select.
      if (englishOnly) delete opts.task;

      // Use the pre-tokenized prompt cache populated by setPrompt messages.
      // Skip for Moonshine (cached IDs are null in that case anyway).
      if (cachedPromptIds && cachedPromptIds.length > 0 && !isMoonshineModel(loadedModelId)) {
        opts.prompt_ids = cachedPromptIds;
      }

      // `await pipe(...)` is the single point where a hang is indistinguishable
      // from slowness from the host's perspective: no result is posted, no error
      // is thrown, and the host only knows it dispatched audio that never came
      // back. The watchdog below converts that silence into a running elapsed
      // count, so a stuck inference is visibly stuck rather than merely absent.
      //
      // CAVEAT, measured: this timer does not always fire. A packaged Windows
      // build ran whisper-large-v3-turbo for 65+ seconds on a 1.6s segment with
      // no STILL RUNNING line at all — heavy ONNX inference starves this
      // thread's timers, so the worker cannot report on itself while it is busy.
      // The host's own watchdog (LocalWhisperSTT.armTaskWatchdog) is therefore
      // the authoritative one; this is a best-effort extra.
      const samples = (msg.audio?.length ?? 0);
      const audioSec = (samples / 16000).toFixed(2);
      const t0 = Date.now();
      console.log(
        `[WhisperWorker] transcribe START task=${msg.taskId} samples=${samples} (${audioSec}s)` +
        ` lang=${language ?? 'auto-detect'} streaming=${streaming}` +
        ` prompt=${opts.prompt_ids ? opts.prompt_ids.length + ' ids' : 'none'} model=${loadedModelId}`,
      );
      const watchdog = setInterval(() => {
        console.warn(
          `[WhisperWorker] transcribe STILL RUNNING task=${msg.taskId}` +
          ` elapsed=${Date.now() - t0}ms audio=${audioSec}s — inference has not returned`,
        );
      }, 5000);
      if (typeof watchdog.unref === 'function') watchdog.unref();

      let result: any;
      try {
        try {
          result = await pipe(msg.audio, opts);
        } catch (e: any) {
          // A checkpoint we did not know was English-only. Record it so every
          // later segment goes out clean, and retry this one immediately rather
          // than reporting a failure the user can do nothing about.
          if (!isEnglishOnlyRefusal(e)) throw e;
          englishOnlyAtRuntime.add(loadedModelId);
          delete opts.language;
          delete opts.task;
          console.warn(
            `[WhisperWorker] ${loadedModelId} is English-only — dropping language/task and retrying task=${msg.taskId}`,
          );
          result = await pipe(msg.audio, opts);
        }
      } finally {
        clearInterval(watchdog);
      }

      const elapsed = Date.now() - t0;
      const text = result?.text ?? '';
      // realtime factor: >1 means transcription is slower than the speech
      // itself, which is what makes results land after a meeting has ended.
      const rtf = samples > 0 ? (elapsed / (samples / 16)).toFixed(2) : 'n/a';
      console.log(
        `[WhisperWorker] transcribe DONE task=${msg.taskId} elapsed=${elapsed}ms` +
        ` realtimeFactor=${rtf}x chars=${text.length} text="${String(text).slice(0, 80)}"`,
      );
      parentPort!.postMessage({
        type: streaming ? 'partial' : 'result',
        taskId: msg.taskId,
        text,
      });
    } catch (e: any) {
      console.error(`[WhisperWorker] transcribe FAILED task=${msg.taskId}: ${e?.stack ?? e?.message ?? e}`);
      parentPort!.postMessage({
        type: 'error',
        taskId: msg.taskId,
        message: `Transcription failed: ${e.message}`,
      });
    }
  }
});
