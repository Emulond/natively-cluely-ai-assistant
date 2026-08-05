// electron/audio/__tests__/LocalWhisperChannelOff.test.mjs
//
// THE PROBLEM THIS FIXES: Whisper ran one session PER CHANNEL — microphone and
// system audio — with no way to say "I only need one of them". The second
// session is not a small extra cost. It is another full copy of the model
// resident in memory and another thread pool contending for the same cores.
//
// Measured on a 6-core i5-11260H with Whisper Small on both channels:
//
//   realtimeFactor  min 2.1x  median 12.1x  max 31.4x
//   elapsed ms      min 3617  median 18580  max 93839
//   latency · final: p50=13148ms p95=30672ms   (system channel)
//
// and, because inference could not keep up, backpressure discarded 13 of 32
// detected phrases — 41% of the user's speech never reached the model. Someone
// who only reads meeting audio was paying all of that for a channel they never
// looked at.
//
// Run: node --test 'electron/audio/__tests__/LocalWhisperChannelOff.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../../..');
const read = (p) => fs.readFileSync(path.resolve(repoRoot, p), 'utf8');

const mainSrc = read('electron/main.ts');
const ipcSrc = read('electron/ipcHandlers.ts');
const cfgSrc = read('electron/audio/whisper/inferenceConfig.ts');
const panelSrc = read('src/components/LocalWhisperModelPanel.tsx');

describe('a channel can be turned off', () => {
  test('the sentinel is shared, not spelled out at each site', async () => {
    const { CHANNEL_MODEL_OFF, isChannelDisabled } = await import(
      pathToFileURL(path.join(repoRoot, 'dist-electron/electron/audio/whisper/modelManager.js')).href
    );
    assert.equal(CHANNEL_MODEL_OFF, 'off');
    assert.equal(isChannelDisabled('off'), true);
    assert.equal(isChannelDisabled('OFF'), true, 'case must not matter');
    assert.equal(isChannelDisabled(' off '), true, 'whitespace must not matter');
    assert.equal(isChannelDisabled('Xenova/whisper-small'), false);
    assert.equal(isChannelDisabled(undefined), false);
    assert.equal(isChannelDisabled(''), false, 'unset means "use the global model"');
  });

  test('an off channel creates no STT provider at all', () => {
    // Returning null is the established "no provider" signal — every capture
    // write site is `?.write`, so the audio is simply never handed anywhere.
    assert.match(
      mainSrc,
      /if \(isChannelDisabled\(channelModel\)\) \{[\s\S]{0,500}?return null;/,
      'an off channel must short-circuit before any model id is resolved',
    );
    // And it must happen BEFORE the model is constructed.
    const offIdx = mainSrc.indexOf('if (isChannelDisabled(channelModel))');
    const ctorIdx = mainSrc.indexOf('const lws = new LocalWhisperSTT(modelId)');
    assert.ok(offIdx > 0 && ctorIdx > offIdx, 'the guard must precede construction');
  });

  test('createSTTProvider already declares it may return null', () => {
    // The whole approach rests on this being an established contract rather
    // than a new failure mode introduced here.
    assert.match(
      mainSrc,
      /private createSTTProvider\(speaker: 'interviewer' \| 'user'\): STTProvider \| null/,
    );
  });

  test('startup repair does not undo the choice', () => {
    // The catalog validator resets unknown ids to a safe fallback. Treating
    // 'off' as unknown would silently re-enable a channel the user disabled
    // and hand back exactly the cost they opted out of.
    assert.match(
      mainSrc,
      /if \(raw && !chanOff\(raw\) && !MODEL_CATALOG_IDS\.has\(raw\)\)/,
    );
  });

  test('the IPC validator accepts it as a real value', () => {
    assert.match(
      ipcSrc,
      /const validChannelModel = \(id: string\) => isChannelDisabled\(id\) \|\| MODEL_CATALOG_IDS\.has\(id\)/,
    );
  });

  test('the settings UI offers it, per channel only', () => {
    assert.match(panelSrc, /id: 'off', name: t\('Off — do not transcribe this channel'\)/);
    assert.match(panelSrc, /onChange=\{setMicModel\}\s*\n\s*options=\{channelOptions\}/);
    assert.match(panelSrc, /onChange=\{setSystemModel\}\s*\n\s*options=\{channelOptions\}/);
    // The global selector must NOT offer it: both channels off would be local
    // transcription with nothing to transcribe.
    assert.match(panelSrc, /onChange=\{setGlobalModel\}\s*\n\s*options=\{availableModels\}/);
  });
});

describe('the surviving channel gets the whole CPU budget', () => {
  test('the session count is computed, not hardcoded', () => {
    assert.match(cfgSrc, /export function resolveActiveWhisperChannelCount\(\): number/);
    assert.match(cfgSrc, /return Math\.max\(1, 2 - off\)/);
    assert.match(cfgSrc, /concurrentSessions: resolveActiveWhisperChannelCount\(\)/);
  });

  test('per-channel mode off means both channels count', () => {
    assert.match(
      cfgSrc,
      /if \(!sm\.get\('localWhisperPerChannelEnabled'\)\) return 2/,
      'without per-channel overrides both channels run the global model',
    );
  });

  test('an unreadable settings store assumes two, never one', () => {
    // Guessing high is safe (fewer threads each); guessing low oversubscribes.
    assert.match(cfgSrc, /\} catch \{[\s\S]{0,300}?return 2;/);
  });

  test('the thread budget divides by the count it is given', async () => {
    const { getBoundedOnnxSessionOptions } = await import(
      pathToFileURL(path.join(repoRoot, 'dist-electron/electron/utils/onnxThreadConfig.js')).href
    );
    if (process.platform === 'darwin') return;
    const os = await import('node:os');
    const realCpus = os.default.cpus;
    os.default.cpus = () => new Array(12).fill({ model: 'synthetic', speed: 0 });
    try {
      const one = getBoundedOnnxSessionOptions(1).intraOpNumThreads;
      const two = getBoundedOnnxSessionOptions(2).intraOpNumThreads;
      assert.ok(one > two, `one live session must get more threads (${one} vs ${two})`);
      assert.ok(one * 1 <= 12 && two * 2 <= 12, 'neither configuration may oversubscribe');
    } finally {
      os.default.cpus = realCpus;
    }
  });

  test('a missing count still behaves like today', async () => {
    const { getBoundedOnnxSessionOptions } = await import(
      pathToFileURL(path.join(repoRoot, 'dist-electron/electron/utils/onnxThreadConfig.js')).href
    );
    assert.equal(
      getBoundedOnnxSessionOptions().intraOpNumThreads,
      getBoundedOnnxSessionOptions(2).intraOpNumThreads,
      'callers that pass nothing must keep the two-session default',
    );
  });

  test('the worker uses the count it was sent for both CPU and GPU sessions', () => {
    const workerSrc = read('electron/audio/whisper/whisperWorker.ts');
    assert.match(workerSrc, /getBoundedOnnxSessionOptions\(msg\.concurrentSessions\)/);
    assert.match(workerSrc, /getDirectMLSessionOptions\(msg\.concurrentSessions\)/);
  });
});

describe('a channel change reaches the running app', () => {
  test('the setting handler rebuilds the live STT providers', () => {
    // THE BUG: providers are built once and reused, so a per-channel change
    // made while the app was running did nothing until the next launch. The
    // setting saved correctly —
    //
    //   [Settings] local Whisper channel models now: splitChannels=true
    //              mic=off system=Xenova/whisper-small
    //
    // — and the meeting ninety seconds later still cold-started TWO workers,
    // because it reused what startup had created. Nothing said the change
    // would not apply, so it looked exactly like the feature not working.
    assert.match(ipcSrc, /await appState\.applyLocalWhisperChannelChange\(\)/);
    assert.match(mainSrc, /public async applyLocalWhisperChannelChange\(\): Promise<void>/);
  });

  test('it reuses the provider-switch path rather than inventing one', () => {
    const body = mainSrc.slice(
      mainSrc.indexOf('public async applyLocalWhisperChannelChange'),
      mainSrc.indexOf('public async reconfigureSttProvider'),
    );
    assert.match(body, /await this\.reconfigureSttProvider\(\)/);
  });

  test('a running meeting is not torn down under the user', () => {
    const body = mainSrc.slice(
      mainSrc.indexOf('public async applyLocalWhisperChannelChange'),
      mainSrc.indexOf('public async reconfigureSttProvider'),
    );
    assert.match(
      body,
      /if \(this\.isMeetingActive\) \{[\s\S]{0,500}?return;/,
      'rebuilding mid-meeting would drop audio; the change waits for the next one',
    );
    assert.match(body, /applies to the next meeting/, 'and it must say so');
  });

  test('a failure to apply cannot fail the save', () => {
    assert.match(
      ipcSrc,
      /try \{\s*\n\s*await appState\.applyLocalWhisperChannelChange\(\);\s*\n\s*\} catch/,
      'the setting is already persisted by this point — applying is best-effort',
    );
  });
});

describe('a stalled worker is visible', () => {
  const workerSrc = read('electron/audio/whisper/whisperWorker.ts');
  const dlSrc = read('electron/services/LocalModelDownloadService.ts');

  test('the worker logs on arrival, before the first slow call', () => {
    // loadTransformers() is a dynamic ESM import of a very large package plus
    // the native ORT binding, and it was the FIRST statement in the init
    // handler — so a worker stalling there produced nothing at all. Two
    // workers, seventy seconds, not one line between them, while the host
    // showed `worker=true ready=false` and a climbing pending queue.
    const initIdx = workerSrc.indexOf("if (msg.type === 'init')");
    const logIdx = workerSrc.indexOf('init received for', initIdx);
    const importIdx = workerSrc.indexOf('await loadTransformers()', initIdx);
    assert.ok(logIdx > initIdx, 'the worker must announce that init arrived');
    assert.ok(importIdx > logIdx, 'that line must come BEFORE the import');
  });

  test('the import duration is reported, so a slow one is measurable', () => {
    assert.match(workerSrc, /transformers imported in \$\{Date\.now\(\) - importT0\}ms/);
  });

  test('a background download reports more than its own beginning', () => {
    // "starting background download" was the first and last line about it.
    assert.match(dlSrc, /worker spawned, downloading\.\.\./);
    assert.match(dlSrc, /complete — files verified on disk/);
  });
});

describe('backlogged speech is merged, not discarded', () => {
  const sttSrc = read('electron/audio/LocalWhisperSTT.ts');

  test('a saturated worker coalesces instead of dropping', () => {
    // Whisper's encoder has a FIXED 30-second receptive field: every clip is
    // padded to 30s before encoding, so a 0.8s phrase costs the same pass as a
    // 25s one. Measured on a 6-core i5-11260H, whisper-small, 10 threads:
    //
    //   "Проверка."                    8395ms  (10.0x realtime)
    //   "Только что-то выразилось..."  6120ms  ( 2.7x realtime)
    //
    // Nearly the same wall time for very different amounts of speech. Dropping
    // a phrase therefore saves NOTHING structural — the next one still pays a
    // whole pass. Appending it to the clip already waiting is free.
    assert.match(sttSrc, /this\.coalesceFinal\(audio\);\s*\n\s*return;/);
    assert.ok(
      !/dropping a final segment/.test(sttSrc),
      'the unconditional drop is what threw away 41% of speech',
    );
  });

  test('the merged clip stays inside the 30s window', () => {
    assert.match(sttSrc, /MAX_COALESCED_SAMPLES = 25 \* 16000/);
    assert.match(
      sttSrc,
      /while \(this\.coalesceSamples > LocalWhisperSTT\.MAX_COALESCED_SAMPLES/,
      'a clip longer than the window cannot be encoded in one pass',
    );
  });

  test('past the window the oldest audio goes, and it is reported', () => {
    // Only here is speech genuinely lost, and it must say so.
    assert.match(sttSrc, /merged backlog exceeded/);
    assert.match(sttSrc, /this\.diagDroppedBackpressure\+\+/);
  });

  test('a freed slot flushes the held audio', () => {
    assert.match(
      sttSrc,
      /this\.inFlightTranscribes = Math\.max\(0, this\.inFlightTranscribes - 1\);\s*\n[\s\S]{0,300}?this\.flushCoalesced\(\);/,
      'held speech must go out as soon as there is room',
    );
  });

  test('the flush respects the same occupancy limit as a normal final', () => {
    const body = sttSrc.slice(
      sttSrc.indexOf('private flushCoalesced()'),
      sttSrc.indexOf('private sendTranscribe('),
    );
    assert.match(body, /if \(outstandingFinals >= LocalWhisperSTT\.MAX_INFLIGHT_TRANSCRIBES\) return/);
    assert.match(body, /if \(this\.coalesceBuffer\.length === 0 \|\| !this\.worker \|\| !this\.workerReady\) return/);
  });

  test('stopping a meeting sends the held audio rather than binning it', () => {
    assert.match(
      sttSrc,
      /segs\.forEach\(s => this\.dispatchFinal\(s\.samples\)\);\s*\n[\s\S]{0,300}?this\.flushCoalesced\(\);/,
    );
  });

  test('a dead worker clears the buffer so it cannot leak into the next session', () => {
    const clears = sttSrc.match(/this\.coalesceBuffer = \[\];\s*\n\s*this\.coalesceSamples = 0;/g) ?? [];
    assert.ok(clears.length >= 2, 'both the worker error and exit paths must clear it');
  });
});

describe('the GPU precision is part of a complete download', () => {
  const svcSrc = read('electron/services/LocalModelDownloadService.ts');
  const mmSrc = read('electron/audio/whisper/modelManager.ts');
  const workerSrc = read('electron/audio/whisper/whisperWorker.ts');

  test('a model missing its GPU precision does not count as cached', () => {
    // THE TRAP: the settings panel renders Install only while a model is NOT
    // available — `{!isAvailable && !isDownloading && (<button>Install</button>)}`
    // — and start() short-circuits for a model reported cached. So once the
    // primary files landed, the model was marked available, the button
    // disappeared, and the fp16 files could never be fetched by ANY route.
    // After several deliberate re-downloads the folder still held only:
    //   encoder_model.onnx 336.5MB, decoder_model_merged_quantized.onnx 149.5MB
    assert.match(
      svcSrc,
      /if \(extra && extra\.length > 0 && !mm\.isGpuPrecisionCached\(modelId as any\)\) return false;/,
    );
  });

  test('the panel really does hide Install for an available model', () => {
    // The premise above, pinned — if this ever changes, the coupling above
    // stops being load-bearing and should be revisited.
    assert.match(panelSrc, /\{!isAvailable && !isDownloading && \(/);
  });

  test('it checks both decoder layouts and either is enough', () => {
    assert.match(mmSrc, /export function isGpuPrecisionCached/);
    assert.match(
      mmSrc,
      /return has\('decoder_model_merged_fp16\.onnx'\)\s*\n\s*\|\| \(has\('decoder_model_fp16\.onnx'\) && has\('decoder_with_past_model_fp16\.onnx'\)\)/,
    );
    assert.match(mmSrc, /if \(!has\('encoder_model_fp16\.onnx'\)\) return false/);
  });

  test('a model with no fp16 build is not stranded forever', () => {
    // Demanding a file that does not exist would leave the model permanently
    // "not downloaded", re-attempting the same missing fetch every launch.
    assert.match(mmSrc, /NO_GPU_PRECISION_MARKER = '\.no-fp16'/);
    assert.match(
      mmSrc,
      /if \(fs\.existsSync\(path\.join\(onnxDir, NO_GPU_PRECISION_MARKER\)\)\) return true;/,
    );
    // And the worker writes it when the fetch proves impossible.
    assert.match(workerSrc, /_pathMark\.join\(dir, '\.no-fp16'\)/);
  });

  test('machines with no usable GPU are unaffected', () => {
    // resolveExtraDownloadDtypes returns undefined without a probed adapter,
    // so the extra requirement never applies and nothing extra is demanded.
    assert.match(svcSrc, /const extra = cfg\.resolveExtraDownloadDtypes\(gpuDeviceIdForCacheCheck\(\)\)/);
    assert.match(
      cfgSrc,
      /if \(gpuDeviceId === null \|\| gpuDeviceId === undefined\) return undefined/,
    );
  });
});
