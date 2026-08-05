// electron/audio/whisper/__tests__/GpuProbeSafety.test.mjs
//
// THE BUGS THIS PINS — two GPU attempts, two crashed apps:
//
//   1. session options. ONNX Runtime documents that the DirectML provider
//      requires memory-pattern optimisation OFF and sequential execution, "or
//      an error will be returned". The shared session options turn memory
//      patterns ON (they help CPU throughput) and were handed straight to the
//      DirectML session. Every GPU attempt was invalid before it began.
//
//   2. where the attempt happened. Both attempts created the DirectML session
//      inside the app's own process. A DirectML failure is a native abort — no
//      JS exception, no fallback, no log — so the app died at launch and a
//      sentinel file written moments earlier could lose the race to disk. The
//      user was left with an app that would not open.
//
// The fix for (1) is getDirectMLSessionOptions(). The fix for (2) is
// structural: the attempt runs in a throwaway child process whose death the
// parent merely observes.
//
// Run: node --test 'electron/audio/whisper/__tests__/GpuProbeSafety.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../../../..');
const read = (p) => fs.readFileSync(path.resolve(repoRoot, p), 'utf8');

describe('DirectML session options', () => {
  test('memory pattern is OFF — ORT rejects the session otherwise', async () => {
    const { getDirectMLSessionOptions } = await import(
      pathToFileURL(path.join(repoRoot, 'dist-electron/electron/utils/onnxThreadConfig.js')).href
    );
    assert.equal(getDirectMLSessionOptions().enableMemPattern, false);
  });

  test('execution is sequential — parallel mode is unsupported by DirectML', async () => {
    const { getDirectMLSessionOptions } = await import(
      pathToFileURL(path.join(repoRoot, 'dist-electron/electron/utils/onnxThreadConfig.js')).href
    );
    assert.equal(getDirectMLSessionOptions().executionMode, 'sequential');
  });

  test('the CPU defaults it diverges from are still memory-pattern ON off macOS', async () => {
    // If this ever stops being true the DirectML override becomes a no-op and
    // the crash quietly returns, so the divergence is the thing worth pinning.
    const { getBoundedOnnxSessionOptions } = await import(
      pathToFileURL(path.join(repoRoot, 'dist-electron/electron/utils/onnxThreadConfig.js')).href
    );
    if (process.platform === 'darwin') return;
    assert.equal(
      getBoundedOnnxSessionOptions().enableMemPattern,
      true,
      'CPU sessions keep memory patterns; DirectML must override them',
    );
  });

  test('the worker builds GPU pipelines with the DirectML options, not the CPU ones', () => {
    const src = read('electron/audio/whisper/whisperWorker.ts');
    assert.match(
      src,
      /const gpuSessionOptions = useGpu \? getDirectMLSessionOptions\(\) : null/,
      'the GPU path must derive its own session options',
    );
    assert.match(
      src,
      /session_options: device && gpuExecutionProviders && gpuSessionOptions\s*\n\s*\? \{ \.\.\.gpuSessionOptions,/,
      'spreading the CPU sessionOptions into a DirectML session is the original crash',
    );
  });
});

describe('the probe cannot take the app down', () => {
  test('the worker no longer chooses an adapter for itself', () => {
    const src = read('electron/audio/whisper/whisperWorker.ts');
    assert.ok(
      !/NATIVELY_ONNX_GPU_LADDER/.test(src),
      'the in-process adapter ladder is what crashed at launch — it must be gone',
    );
    assert.match(
      src,
      /typeof msg\.gpuDeviceId === 'number' \? msg\.gpuDeviceId : null/,
      'the adapter must arrive as a verdict from the probe, not a local guess',
    );
  });

  test('the probe runs as a separate process, not a worker thread', () => {
    const src = read('electron/audio/whisper/gpuProbe.ts');
    assert.match(src, /spawn\(\s*process\.execPath/, 'must spawn a real child process');
    assert.match(src, /ELECTRON_RUN_AS_NODE: '1'/);
    // A worker_thread shares the process: a native abort there is still fatal.
    assert.ok(!/new Worker\(/.test(src), 'a worker thread would not survive a native abort');
  });

  test('a child that dies by signal is a failure, not a crash', () => {
    const src = read('electron/audio/whisper/gpuProbe.ts');
    assert.match(src, /child\.on\('exit'/);
    assert.match(src, /signal\s*\n?\s*\? `killed by \$\{signal\}`/);
  });

  test('a hung probe is bounded — a wedged driver must not stall startup', () => {
    const src = read('electron/audio/whisper/gpuProbe.ts');
    assert.match(src, /PROBE_TIMEOUT_MS/);
    assert.match(src, /child\.kill\('SIGKILL'\)/);
  });

  test('exit 0 is not accepted as proof the GPU ran', () => {
    // ORT silently drops an unavailable provider and builds a CPU session that
    // exits 0. Trusting that reports acceleration on machines that have none.
    const src = read('electron/audio/whisper/gpuProbe.ts');
    assert.match(src, /removing requested execution provider/i);
  });

  test('every failure path resolves to CPU rather than throwing', () => {
    const src = read('electron/audio/whisper/gpuProbe.ts');
    assert.match(src, /const cpu = \(reason: string\): GpuProbeResult/);
    assert.match(src, /ensureGpuProbe[\s\S]{0,600}?\.catch\(/, 'the memo must never reject');
  });

  test('discrete adapters are probed before integrated ones', () => {
    const src = read('electron/audio/whisper/gpuProbe.ts');
    assert.match(src, /VENDOR_NVIDIA = 0x10de/);
    assert.match(src, /VENDOR_AMD = 0x1002/);
    assert.match(src, /candidates\.sort\(\(a, b\) => Number\(b\.discrete\) - Number\(a\.discrete\)\)/);
  });

  test('the verdict is cached against GPU, driver and ORT version', () => {
    const src = read('electron/audio/whisper/gpuProbe.ts');
    assert.match(src, /driverVersion/);
    assert.match(src, /ort\$\{ortVersion\}/);
    assert.match(src, /parsed\?\.signature !== signature/, 'a stale signature must re-probe');
  });
});

describe('the probe child itself', () => {
  const child = path.join(repoRoot, 'dist-electron/electron/audio/whisper/gpuProbeChild.js');

  test('is built as a standalone script', () => {
    assert.ok(fs.existsSync(child), 'gpuProbeChild.js must exist in dist-electron');
  });

  test('exits non-zero and reports JSON when it cannot open the model', () => {
    let code = 0;
    let out = '';
    try {
      out = execFileSync(process.execPath, [child, '/nonexistent/model.onnx', '0'], {
        encoding: 'utf8', timeout: 60000, stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch (e) {
      code = e.status ?? -1;
      out = String(e.stdout ?? '');
    }
    assert.notEqual(code, 0, 'a failed probe must not report success');
    assert.match(out, /"ok":false/);
  });

  test('exits 2 when given no model path', () => {
    let code = 0;
    try {
      execFileSync(process.execPath, [child], {
        encoding: 'utf8', timeout: 60000, stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch (e) {
      code = e.status ?? -1;
    }
    assert.equal(code, 2);
  });

  test('is unpacked from the asar — a .node addon cannot load from an archive', () => {
    const pkg = JSON.parse(read('package.json'));
    assert.ok(
      pkg.build.asarUnpack.includes('**/gpuProbeChild.js'),
      'the probe requires onnxruntime-node, so it must live outside app.asar',
    );
  });
});

describe('external-data stubs are not probed', () => {
  test('files under 1MB are skipped when picking a probe model', async () => {
    // A 0.4MB graph stub whose weights live in a sibling .onnx_data fails to
    // open for reasons that have nothing to do with the GPU — probing one would
    // blame DirectML for a missing file.
    const { findSmallestCachedOnnx } = await import(
      pathToFileURL(path.join(repoRoot, 'dist-electron/electron/audio/whisper/gpuProbe.js')).href
    );
    const dir = fs.mkdtempSync(path.join(process.env.TMPDIR ?? '/tmp', 'gpuprobe-'));
    fs.writeFileSync(path.join(dir, 'stub.onnx'), Buffer.alloc(400_000));
    assert.equal(findSmallestCachedOnnx(dir), null, 'a stub alone must yield no candidate');

    const real = path.join(dir, 'encoder_model.onnx');
    fs.writeFileSync(real, Buffer.alloc(2_000_000));
    assert.equal(findSmallestCachedOnnx(dir), real);
  });
});

describe('int8 is proven separately from float', () => {
  const probeSrc = read('electron/audio/whisper/gpuProbe.ts');
  const workerSrc = read('electron/audio/whisper/whisperWorker.ts');

  test('the probe tests a quantised graph after the fp32 one', () => {
    // THE CRASH: the probe cleared a GTX 1650 on the fp32 encoder in 1033ms,
    // then the app died one line after "building pipeline with device=dml
    // deviceId=1" — when transformers.js built the q8 DECODER session. No
    // exception, no exit code. DirectML's int8 operator coverage is narrower
    // than its float coverage, and the probe was only testing float.
    assert.match(probeSrc, /export function findQuantizedOnnx/);
    assert.match(probeSrc, /const q8 = await runProbeChild\(q8Path, deviceId\)/);
    assert.match(probeSrc, /quantizedOk = q8\.ok/);
  });

  test('a refusal downgrades the dtype, it does not abandon the GPU', () => {
    // fp32 on a Turing card still beats the CPU comfortably.
    assert.match(workerSrc, /const gpuRejectsInt8 = useGpu && msg\.gpuQuantizedOk !== true/);
    assert.match(workerSrc, /effectiveDtype = usable/, 'fp16 or fp32, whichever is cached');
    assert.match(workerSrc, /dtype: device \? effectiveDtype : dtype/);
  });

  test('the replacement precision is only chosen when it is already on disk', () => {
    // Models download at the mixed default — fp32 encoder, QUANTISED decoders.
    // Asking for any other uniform precision asks for files that were never
    // fetched, and transformers.js answers by downloading them, silently, at
    // meeting start. The worker never reaches "model READY" and the channel
    // shows "STT reconnecting" forever. The diagnostics said so all along:
    //   decoderFile: decoder_model_merged.onnx  decoderBytes: -1
    assert.match(workerSrc, /const dtypeFilesPresent = \(dt: 'fp16' \| 'fp32'\): boolean =>/);
    assert.match(
      workerSrc,
      /const usable = \(\['fp16', 'fp32'\] as const\)\.find\(dtypeFilesPresent\)/,
      'fp16 must be preferred: half the download, half the VRAM, native to DirectML',
    );
    assert.match(
      workerSrc,
      /\} else \{[\s\S]{0,700}?preferredDevice = undefined;/,
      'with neither precision cached the GPU must be dropped, not a download started',
    );
  });

  test('the presence check accepts either decoder layout', () => {
    // A model ships EITHER a merged decoder OR the split decoder + with_past
    // pair; demanding both would reject every model.
    assert.match(
      workerSrc,
      /return has\(`decoder_model_merged\$\{suffix\}\.onnx`\)\s*\n\s*\|\| \(has\(`decoder_model\$\{suffix\}\.onnx`\) && has\(`decoder_with_past_model\$\{suffix\}\.onnx`\)\)/,
    );
  });

  test('downloads fetch the GPU precision up front, meetings never do', () => {
    // The one moment the user has deliberately asked to wait for bytes.
    const cfg = read('electron/audio/whisper/inferenceConfig.ts');
    assert.match(cfg, /export function resolveExtraDownloadDtypes/);
    assert.match(cfg, /return \['fp16'\]/);
    assert.match(
      cfg,
      /extraDtypes: opts\?\.forDownload \? resolveExtraDownloadDtypes\(gpuDeviceId\) : undefined/,
      'extraDtypes must be opt-in per call, never set on the meeting path',
    );
    assert.match(
      cfg,
      /if \(gpuDeviceId === null \|\| gpuDeviceId === undefined\) return undefined/,
      'a machine with no usable GPU must not download bytes it can never read',
    );
    assert.match(
      read('electron/services/LocalModelDownloadService.ts'),
      /buildWorkerInitMessage\(modelId, \{ forDownload: true \}\)/,
    );
  });

  test('the extra fetch completes before the download reports ready', () => {
    // The service treats 'ready' as completion; posting it first would report a
    // finished download with files still arriving.
    const loopIdx = workerSrc.indexOf('for (const extra of msg.extraDtypes');
    const readyIdx = workerSrc.indexOf("postMessage({ type: 'ready' })", loopIdx);
    assert.ok(loopIdx > 0, 'the extra-precision loop must exist');
    assert.ok(readyIdx > loopIdx, "'ready' must be posted after the extra passes");
  });

  test('an unpublished precision does not fail an otherwise good download', () => {
    assert.match(
      workerSrc,
      /catch \(extraErr: any\) \{[\s\S]{0,500}?the model itself downloaded fine/,
    );
  });

  test('the CPU path keeps its mixed q8 dtype', () => {
    // q8 decoders are the main CPU speed win and are not implicated here.
    assert.match(
      workerSrc,
      /dtype: device \? effectiveDtype : dtype/,
      'a CPU build must pass the original dtype, not the GPU downgrade',
    );
  });

  test('the probe verdict carries quantizedOk end to end', () => {
    assert.match(probeSrc, /quantizedOk: boolean/);
    assert.match(read('electron/audio/whisper/types.ts'), /gpuQuantizedOk\?: boolean/);
    assert.match(
      read('electron/audio/whisper/inferenceConfig.ts'),
      /gpuQuantizedOk = !!probe\.quantizedOk/,
    );
  });

  test('probe logic version was bumped so the old verdict cannot be reused', () => {
    const version = Number(probeSrc.match(/const PROBE_LOGIC_VERSION = (\d+)/)?.[1]);
    assert.ok(version >= 3, `expected >= 3, got ${version}`);
  });
});

describe('a GPU pipeline build that never returns is not repeated', () => {
  const workerSrc = read('electron/audio/whisper/whisperWorker.ts');

  test('a sentinel is written before the build and cleared after', () => {
    // The probe makes the DECISION safe; it cannot make every consequence safe,
    // because it opens one graph and the pipeline opens several. If a build
    // still aborts, nothing in-process survives to report it — so the record
    // goes down first.
    assert.match(workerSrc, /\.gpu-build-sentinel\.json/);
    // Ordering is the whole point: written BEFORE, cleared AFTER. The window
    // between them holds the "building pipeline with device=..." log line.
    const writeIdx = workerSrc.indexOf('_fsSentinel.writeFileSync(gpuBuildSentinel');
    const buildIdx = workerSrc.indexOf('pipe = await buildPipeline(preferredDevice);');
    const clearIdx = workerSrc.indexOf('clearGpuBuildSentinel();', buildIdx);
    assert.ok(writeIdx > 0, 'the sentinel must be written');
    assert.ok(buildIdx > writeIdx, 'the write must precede the GPU build');
    assert.ok(clearIdx > buildIdx, 'the clear must follow a successful build');
  });

  test('a stale sentinel for the same model forces CPU', () => {
    assert.match(
      workerSrc,
      /if \(preferredDevice && poisonedBuild\?\.modelId === msg\.modelId\) \{[\s\S]{0,600}?preferredDevice = undefined;/,
      'the second attempt at a build that already killed the app must go to CPU',
    );
  });

  test('a caught build error clears it too — that path is recoverable', () => {
    assert.match(
      workerSrc,
      /\} catch \(deviceErr: any\) \{\s*\n\s*clearGpuBuildSentinel\(\);/,
      'a JS throw fell back to CPU in-process; it must not poison the next launch',
    );
  });
});
