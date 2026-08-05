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
import { getBoundedOnnxSessionOptions, getDirectMLSessionOptions } from '../../utils/onnxThreadConfig';
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
      // LOG BEFORE THE FIRST SLOW THING, NOT AFTER.
      //
      // loadTransformers() is a dynamic ESM import of a very large package plus
      // the native ONNX Runtime binding, and it was the first statement in this
      // handler — so a worker that stalled there produced NOTHING. No "Loading",
      // no error, no trace of any kind. The host saw only `worker=true
      // ready=false` forever and a pending queue climbing:
      //
      //   [LocalWhisperSTT] Cold-starting worker for Xenova/whisper-small
      //   [LocalWhisperSTT:system] pipeline · pending=7 ... worker=true ready=false
      //
      // Two workers, seventy seconds, not one line between them. Every
      // "the worker never became ready and said nothing" report traces back to
      // this ordering. An arrival line costs nothing and makes the difference
      // between "stalled importing transformers" and "stalled loading weights"
      // visible without guessing.
      console.log(
        `[WhisperWorker] init received for ${msg.modelId} — importing @huggingface/transformers...`,
      );
      const importT0 = Date.now();
      const { pipeline, env } = await loadTransformers();
      console.log(`[WhisperWorker] transformers imported in ${Date.now() - importT0}ms`);

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
      const sessionOptions = getBoundedOnnxSessionOptions(msg.concurrentSessions);

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

        // FULL ON-DISK INVENTORY.
        //
        // The two lines above report only the files ONE dtype expects, which is
        // how an entire debugging round got spent arguing about whether a model
        // was downloaded. It was — just not in the precision that had been asked
        // for. Whisper ships each module at several precisions, distinguished by
        // a filename suffix ('' = fp32, '_quantized' = q8, '_fp16' = fp16), and a
        // normal download fetches an fp32 encoder with q8 decoders. Nothing
        // printed the actual directory, so "downloaded" and "downloaded in the
        // precision we are about to request" looked identical in the log.
        //
        // Listing the directory answers that in one line, offline, without
        // anyone needing to reach the model host to find out what it publishes.
        const DTYPE_SUFFIX: Record<string, string> = {
          fp32: '', fp16: '_fp16', int8: '_int8', uint8: '_uint8',
          q8: '_quantized', q4: '_q4', q4f16: '_q4f16', bnb4: '_bnb4',
        };
        const _mb = (bytes: number) => (bytes < 0 ? null : Math.round(bytes / 1048576 * 10) / 10);
        let _onDisk: Array<{ file: string; mb: number | null }> = [];
        try {
          _onDisk = (_fs.readdirSync(_modelDir) as string[])
            .filter((f) => f.endsWith('.onnx') || f.endsWith('.onnx_data'))
            .sort()
            .map((f) => ({ file: f, mb: _mb(_stat(_path.join(_modelDir, f))) }));
        } catch { /* directory unreadable — the `present` list stays empty */ }

        // What the dtype actually in force will ask the loader to open.
        const _dtypeFor = (module: string): string =>
          typeof dtype === 'string' ? dtype : ((dtype as Record<string, string>)[module] ?? 'fp32');
        const _wanted = ['encoder_model', 'decoder_model_merged', 'decoder_model', 'decoder_with_past_model']
          .map((module) => {
            const dt = _dtypeFor(module);
            const file = `${module}${DTYPE_SUFFIX[dt] ?? ''}.onnx`;
            const bytes = _stat(_path.join(_modelDir, file));
            return { module, dtype: dt, file, present: bytes > 0, mb: _mb(bytes) };
          });
        const _missing = _wanted.filter((w) => !w.present).map((w) => w.file);

        console.log('[WhisperWorker][diag:files] ' + JSON.stringify({
          modelDir: _modelDir,
          present: _onDisk,
          requiredForThisDtype: _wanted,
          // Whisper models ship EITHER a merged decoder OR the split pair, so a
          // missing entry is only a real problem when neither layout is complete.
          note: 'a model provides either decoder_model_merged OR decoder_model + decoder_with_past_model',
          missing: _missing,
        }));
        if (_missing.length > 0) {
          console.warn(
            `[WhisperWorker] ${msg.modelId}: these files are NOT on disk for the requested ` +
            `precision — ${_missing.join(', ')}. If the loader needs one of them it will try ` +
            'to DOWNLOAD it now, which can stall model load for minutes with no other symptom.',
          );
        }
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
      // THE WORKER NO LONGER DECIDES. Two earlier designs picked an adapter
      // here — first `device: 'dml'`, then an automatic ladder — and both took
      // the app down at launch. A DirectML failure is a native abort: no JS
      // throw, so no try/catch and no fallback, and a sentinel written moments
      // earlier can lose the race to disk. The decision now happens in a
      // throwaway child process before any meeting starts (gpuProbe.ts), and
      // this worker is told the answer. Only an adapter that has already
      // survived a real session creation on this machine ever arrives here.
      //
      // WHY THOSE ATTEMPTS WERE DOOMED ANYWAY: ONNX Runtime requires memory
      // pattern optimisation OFF and sequential execution for DirectML "or an
      // error will be returned", and the shared session options enable memory
      // patterns for CPU throughput. Every attempt was invalid before it began.
      // getDirectMLSessionOptions() is the corrected set.
      //
      // HOW THIS REACHES ORT. Passing transformers.js a `device` string gives no
      // way to specify deviceId. But it assigns its own provider list with
      //     session_options.executionProviders ??= executionProviders;
      // — `??=`, so a list we supply is preserved. We therefore build the EP
      // list ourselves and pass it through session_options, with 'cpu' appended
      // so ORT falls back per-operator for anything DirectML cannot run.
      const probedDeviceId: number | null =
        typeof msg.gpuDeviceId === 'number' ? msg.gpuDeviceId : null;
      const gpuReason: string = msg.gpuReason ?? 'no GPU decision supplied';

      // NATIVELY_ONNX_DEVICE=cpu still forces CPU, for support cases.
      const forcedCpu = (process.env.NATIVELY_ONNX_DEVICE ?? '').trim().toLowerCase() === 'cpu';
      const useGpu = probedDeviceId !== null && !forcedCpu;

      console.log(
        `[WhisperWorker] GPU decision: ${useGpu ? `DirectML adapter ${probedDeviceId}` : 'CPU'}` +
        `${forcedCpu ? ' (forced by NATIVELY_ONNX_DEVICE=cpu)' : ''} — ${gpuReason}`,
      );

      // DirectML's own mandatory session options — NOT the CPU ones. Passing
      // the CPU set here is precisely the bug that crashed the app twice.
      const gpuSessionOptions = useGpu ? getDirectMLSessionOptions(msg.concurrentSessions) : null;
      let preferredDevice: string | undefined = useGpu ? 'dml' : undefined;

      // INT8 IS NOT A GIVEN ON DIRECTML.
      //
      // Whisper ships a quantised decoder and an fp32 encoder. A GTX 1650 that
      // the probe had just cleared — CPU control fine, DirectML session on the
      // fp32 encoder fine in 1033ms — still took the whole app down one line
      // after "building pipeline with device=dml deviceId=1", when
      // transformers.js went on to build the q8 decoder session. No exception,
      // no exit code: the boot marker recorded main.ts starting and never
      // reaching its exit handler.
      //
      // DirectML's int8 operator coverage is narrower than its float coverage,
      // so the probe now tests a quantised graph separately. When the adapter
      // refuses one, the GPU is still worth using — just at fp32 throughout,
      // which is also where a Turing card is fastest anyway.
      const gpuRejectsInt8 = useGpu && msg.gpuQuantizedOk !== true;

      // ...AND THE REPLACEMENT PRECISION MUST ALREADY BE ON DISK.
      //
      // Models download at the mixed default: an fp32 encoder and QUANTISED
      // decoders. Asking for any other uniform precision asks for files that
      // were never fetched, and transformers.js answers by downloading them —
      // hundreds of megabytes, silently, at meeting start. The worker never
      // reaches "model READY", the channel shows "STT reconnecting" forever,
      // and nothing in the log says a download is underway. The diagnostics
      // recorded it plainly and it was missed:
      //
      //   encoderFile: encoder_model.onnx         encoderBytes: 352839389
      //   decoderFile: decoder_model_merged.onnx  decoderBytes: -1
      //
      // fp16 FIRST. Both alternatives are published — for Xenova/whisper-small,
      // the fp32 decoder is 615MB against 308MB for fp16, and the fp32 encoder
      // 353MB against 177MB. fp16 is half the disk, half the VRAM (484MB vs
      // 968MB, the difference between comfortable and tight on a 4GB laptop
      // card), DirectML's native precision rather than a tolerated one, and
      // twice the arithmetic rate on Turing and later. fp32 is the fallback for
      // a machine that happens to have those files already.
      const dtypeFilesPresent = (dt: 'fp16' | 'fp32'): boolean => {
        try {
          const _fs = require('fs');
          const _path = require('path');
          const parts = String(msg.modelId).split('/');
          const dir = _path.join(String(msg.cacheDir), parts[0] ?? '', parts[1] ?? '', 'onnx');
          const suffix = dt === 'fp32' ? '' : '_fp16';
          const has = (name: string) => {
            try { return _fs.statSync(_path.join(dir, name)).size > 0; } catch { return false; }
          };
          if (!has(`encoder_model${suffix}.onnx`)) return false;
          // A model ships EITHER a merged decoder OR the split pair.
          return has(`decoder_model_merged${suffix}.onnx`)
            || (has(`decoder_model${suffix}.onnx`) && has(`decoder_with_past_model${suffix}.onnx`));
        } catch { return false; }
      };

      let effectiveDtype: string | Record<string, string> = dtype;
      if (gpuRejectsInt8) {
        const usable = (['fp16', 'fp32'] as const).find(dtypeFilesPresent);
        if (usable) {
          effectiveDtype = usable;
          console.log(
            `[WhisperWorker] DirectML did not accept an int8 graph on this adapter — ` +
            `loading the ${usable} weights already on disk instead of the mixed q8 default`,
          );
        } else {
          console.warn(
            `[WhisperWorker] DirectML refuses int8 on this adapter and ${msg.modelId} has ` +
            'neither fp16 nor fp32 weights cached — using the CPU rather than triggering a ' +
            'multi-hundred-MB download in the middle of a meeting. Re-download this model in ' +
            'Settings → Audio to fetch the GPU precision alongside it.',
          );
          preferredDevice = undefined;
        }
      }

      // Our own EP list wins over transformers.js's, per the ??= above.
      const gpuExecutionProviders = useGpu
        ? [{ name: 'dml', deviceId: probedDeviceId }, 'cpu']
        : undefined;
      const deviceIdInUse = useGpu ? probedDeviceId : undefined;

      const buildPipeline = (device?: string) => pipeline('automatic-speech-recognition', msg.modelId, {
        dtype: device ? effectiveDtype : dtype,
        session_options: device && gpuExecutionProviders && gpuSessionOptions
          ? { ...gpuSessionOptions, executionProviders: gpuExecutionProviders }
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

      // CRASH SENTINEL around the GPU pipeline build.
      //
      // The probe makes the DECISION safe; it cannot make every consequence of
      // that decision safe, because the probe opens one graph and the pipeline
      // opens several. If a build still aborts natively here, nothing in this
      // process survives to record it — so the record is written FIRST and
      // cleared on success. A stale sentinel at the next launch means the last
      // GPU attempt never returned, and that model goes to the CPU instead of
      // crashing the app a second time.
      //
      // Unlike the startup-time sentinel this replaces, this write happens well
      // after boot with the filesystem long since warm, so it is not racing the
      // abort from a standing start.
      const _fsSentinel = require('fs');
      const _pathSentinel = require('path');
      const gpuBuildSentinel = _pathSentinel.join(
        String(msg.cacheDir ?? '.'), '.gpu-build-sentinel.json',
      );
      const readGpuBuildSentinel = (): any | null => {
        try { return JSON.parse(_fsSentinel.readFileSync(gpuBuildSentinel, 'utf8')); } catch { return null; }
      };
      const clearGpuBuildSentinel = (): void => {
        try { _fsSentinel.unlinkSync(gpuBuildSentinel); } catch { /* absent is fine */ }
      };

      const poisonedBuild = readGpuBuildSentinel();
      if (preferredDevice && poisonedBuild?.modelId === msg.modelId) {
        console.warn(
          `[WhisperWorker] SKIPPING GPU for ${msg.modelId}: a previous GPU pipeline build ` +
          `never completed (${JSON.stringify(poisonedBuild)}). Using CPU. ` +
          `Delete ${gpuBuildSentinel} to try again.`,
        );
        preferredDevice = undefined;
      }

      try {
        if (preferredDevice) {
          try {
            _fsSentinel.writeFileSync(gpuBuildSentinel, JSON.stringify({
              modelId: msg.modelId, deviceId: deviceIdInUse, dtype: effectiveDtype, at: Date.now(),
            }));
          } catch { /* best-effort */ }
          console.log(
            `[WhisperWorker] building pipeline with device=${preferredDevice} deviceId=${deviceIdInUse ?? '(ORT default)'}` +
            ` providers=${JSON.stringify(gpuExecutionProviders)}` +
            ` sessionOptions=${JSON.stringify(gpuSessionOptions)}`,
          );
        } else {
          console.log('[WhisperWorker] building pipeline on CPU (library default device)');
        }
        pipe = await buildPipeline(preferredDevice);
        // Survived the build — this configuration is safe to use again.
        clearGpuBuildSentinel();
      } catch (deviceErr: any) {
        clearGpuBuildSentinel();
        // The probe proved the adapter can create a session, but the probe used
        // one small model — a different checkpoint can still be rejected (an
        // operator DirectML cannot run, or weights too large for the card's
        // VRAM). That arrives as an ordinary JS throw, so CPU is still reachable
        // and transcription continues slowly instead of the channel having no
        // worker at all, which is silent from the user's side.
        if (!preferredDevice) throw deviceErr;
        console.warn(
          `[WhisperWorker] device=${preferredDevice} failed to initialise for ${msg.modelId} ` +
          `(${deviceErr?.message ?? deviceErr}) — falling back to CPU`,
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

      // EXTRA PRECISIONS — download path only (see WorkerInitMessage.extraDtypes).
      //
      // Deliberately BEFORE 'ready': the download service treats ready as
      // completion, so posting first would report a finished download with
      // files still arriving. Each pass is best-effort — a precision a
      // repository does not publish must not fail a download that otherwise
      // succeeded, so the failure is logged and the loop moves on.
      for (const extra of msg.extraDtypes ?? []) {
        const label = typeof extra === 'string' ? extra : JSON.stringify(extra);
        try {
          console.log(`[WhisperWorker] fetching extra precision ${label} for ${msg.modelId}...`);
          const t0 = Date.now();
          await pipeline('automatic-speech-recognition', msg.modelId, {
            dtype: extra,
            session_options: sessionOptions,
            ...(useExternalDataFormat !== undefined
              ? { use_external_data_format: useExternalDataFormat }
              : {}),
            progress_callback: (data: any) => {
              const { pct } = aggregator.update(data);
              if (pct === null) return;
              parentPort!.postMessage({ type: 'progress', modelId: msg.modelId, progress: pct });
            },
          });
          console.log(
            `[WhisperWorker] extra precision ${label} cached for ${msg.modelId} in ${Date.now() - t0}ms`,
          );
        } catch (extraErr: any) {
          console.warn(
            `[WhisperWorker] extra precision ${label} unavailable for ${msg.modelId} ` +
            `(${extraErr?.message ?? extraErr}) — the model itself downloaded fine; ` +
            'GPU inference will use whichever precision is present.',
          );
          // Record it, so "downloaded" stops demanding a file that does not
          // exist. Without this the model would read as incomplete forever and
          // re-attempt the same missing fetch on every launch.
          try {
            const _fsMark = require('fs');
            const _pathMark = require('path');
            const parts = String(msg.modelId).split('/');
            const dir = _pathMark.join(String(msg.cacheDir), parts[0] ?? '', parts[1] ?? '', 'onnx');
            _fsMark.mkdirSync(dir, { recursive: true });
            _fsMark.writeFileSync(
              _pathMark.join(dir, '.no-fp16'),
              'This model has no fp16 build; GPU inference will use another precision.\n',
            );
          } catch { /* best-effort — the cost is one retry next launch */ }
        }
      }

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
