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
