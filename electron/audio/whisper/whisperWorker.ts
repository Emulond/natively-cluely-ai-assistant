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

// Distil-Whisper checkpoints have NO multilingual decoder. If the user picks
// 'auto' or any non-English language, the worker will silently transcribe
// non-English audio as phonetic English. Force language='english' so the
// behaviour is at least documented and consistent.
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
      const SUPPORTED_DEVICES = new Set(['cpu', 'dml', 'cuda', 'webgpu', 'gpu', 'auto']);
      const preferredDevice = providers.find((p) => SUPPORTED_DEVICES.has(p));

      const buildPipeline = (device?: string) => pipeline('automatic-speech-recognition', msg.modelId, {
        dtype,
        session_options: sessionOptions,
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
        console.log(`[WhisperWorker] building pipeline with device=${preferredDevice ?? '(library default)'}`);
        pipe = await buildPipeline(preferredDevice);
      } catch (deviceErr: any) {
        // A GPU provider can fail for reasons entirely outside our control —
        // no compatible adapter, a driver that rejects the graph, a DirectML
        // version mismatch. Falling back to CPU keeps transcription working
        // (slowly) instead of leaving the channel with no worker at all, which
        // is silent from the user's side.
        if (!preferredDevice || preferredDevice === 'cpu') throw deviceErr;
        console.warn(
          `[WhisperWorker] device=${preferredDevice} failed to initialise (${deviceErr?.message ?? deviceErr}) — falling back to CPU`,
        );
        pipe = await buildPipeline('cpu');
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
          deviceUsed: preferredDevice ?? '(library default)',
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

      // English-only checkpoints (Distil-Whisper + .en variants) have no
      // multilingual decoder. Force language='english' regardless of the
      // user's auto/non-English setting so the model isn't asked to
      // transcribe phonetically into the wrong language.
      if (ENGLISH_ONLY_MODELS.has(loadedModelId)) {
        language = 'english';
      }

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
        result = await pipe(msg.audio, opts);
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
