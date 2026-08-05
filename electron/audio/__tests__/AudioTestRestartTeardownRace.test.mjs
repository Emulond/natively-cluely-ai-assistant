// electron/audio/__tests__/AudioTestRestartTeardownRace.test.mjs
//
// THE BUG THIS PINS: restarting the audio test reopened the input device while
// the previous capture's NATIVE teardown was still queued.
//
// MicrophoneCapture.stop() defers the native stop to a setImmediate and returns
// a promise for it. Its own documentation says callers should await that
// "before constructing a new native instance / starting again". The audio-test
// start path called stopAudioTest() — which fires stop() and discards the
// promise — and then immediately constructed and started a new capture on the
// same device.
//
// Scrolling the Settings → Audio tab does exactly that, repeatedly, as the level
// meter moves in and out of view. Two crash logs end on the identical three
// lines, ~2s before the process disappeared:
//
//   [Main] Stopping Audio Test
//   [MicrophoneCapture] Stopping capture (deferred native teardown)...
//   (~1s) Starting Audio Test → [MicrophoneCapture] Starting native capture...
//
// and then nothing. No exception, no exit code — the boot marker records main.ts
// being evaluated and never reaching its exit handler, which is what a fault
// inside the native audio backend looks like from JavaScript. Two input streams
// on one endpoint, one of them being freed.
//
// Run: node --test 'electron/audio/__tests__/AudioTestRestartTeardownRace.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../../..');
const read = (p) => fs.readFileSync(path.resolve(repoRoot, p), 'utf8');

const mainSrc = read('electron/main.ts');
const ipcSrc = read('electron/ipcHandlers.ts');

describe('audio test restart waits for the native teardown', () => {
  test('an awaitable teardown helper exists', () => {
    assert.match(
      mainSrc,
      /public async stopAudioTestAndAwaitTeardown\(\): Promise<void>/,
      'stopAudioTest() alone only schedules the teardown',
    );
  });

  test('the teardown promise outlives the wrapper reference', () => {
    // THE FIRST ATTEMPT AT THIS FIX DID NOT WORK, for this exact reason: it
    // read this.audioTestCapture and awaited that wrapper's stop(). But the
    // renderer sends stop-audio-test BEFORE start-audio-test, and stopAudioTest
    // nulls the field — so the start path found nothing to await and sailed
    // through into reopening the device. The crash came back unchanged.
    assert.match(
      mainSrc,
      /private _audioTestTeardown: Promise<unknown> = Promise\.resolve\(\)/,
      'the pending teardown must be held on the instance, not re-derived',
    );
    const body = mainSrc.slice(
      mainSrc.indexOf('public async stopAudioTestAndAwaitTeardown'),
      mainSrc.indexOf('private _audioTestTeardown'),
    );
    assert.match(body, /await this\._audioTestTeardown/);
    // Strip comments first: the surrounding documentation names the field on
    // purpose, to record why the earlier version of this fix did nothing.
    const code = body
      .split('\n')
      .filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line))
      .join('\n');
    assert.ok(
      !/this\.audioTestCapture/.test(code),
      'awaiting a field stopAudioTest() has already nulled is the bug this pins',
    );
  });

  test('stopAudioTest records both wrappers teardown before nulling them', () => {
    // The system probe opens its own native stream and has the same hazard.
    const body = mainSrc.slice(
      mainSrc.indexOf('public stopAudioTest(): void'),
      mainSrc.indexOf('public finalizeMicSTT'),
    );
    assert.match(body, /pending\.push\(Promise\.resolve\(this\.audioTestCapture\.stop\(\)\)\)/);
    assert.match(body, /pending\.push\(Promise\.resolve\(this\.audioTestSystemCapture\.stop\(\)\)\)/);
    assert.match(
      body,
      /this\._audioTestTeardown = Promise\.allSettled\(pending\)/,
      'a rejecting stop must not abort the restart',
    );
  });

  test('starting the test awaits the previous teardown', () => {
    // This is the fix. Anything that goes back to the fire-and-forget call
    // reintroduces the crash.
    assert.match(
      mainSrc,
      /await this\.stopAudioTestAndAwaitTeardown\(\);[^\n]*\n[\s\S]{0,400}?const startEpoch = \+\+this\._audioTestEpoch/,
      '_startAudioTestImpl must await the teardown before opening the device again',
    );
    const startBody = mainSrc.slice(
      mainSrc.indexOf('private async _startAudioTestImpl'),
      mainSrc.indexOf('const startEpoch = ++this._audioTestEpoch'),
    );
    assert.ok(
      !/^\s*this\.stopAudioTest\(\);/m.test(startBody),
      'the un-awaited stopAudioTest() call is what raced the native teardown',
    );
  });

  test('the stop-audio-test IPC resolves only after the teardown', () => {
    // The renderer sends start immediately after stop when the meter scrolls
    // back into view; resolving early hands it a device that is still closing.
    assert.match(
      ipcSrc,
      /safeHandle\('stop-audio-test'[\s\S]{0,600}?await appState\.stopAudioTestAndAwaitTeardown\(\)/,
    );
  });

  test('MicrophoneCapture.stop() still returns the teardown promise', () => {
    // The whole fix rests on this contract.
    const src = read('electron/audio/MicrophoneCapture.ts');
    assert.match(src, /public stop\(\): Promise<void>/);
    assert.match(
      src,
      /return this\._teardownPromise \?\? Promise\.resolve\(\)/,
      'repeated stop() calls must join the in-flight teardown, not start a second',
    );
  });

  test('SystemAudioCapture.stop() does too', () => {
    const src = read('electron/audio/SystemAudioCapture.ts');
    assert.match(src, /public stop\(\): Promise<void>/);
  });
});
