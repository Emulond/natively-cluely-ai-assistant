// electron/utils/__tests__/OnnxThreadDefaults.test.mjs
//
// Regression coverage for the ONNX intra-op thread default.
//
// THE BUG THIS PINS: getBoundedOnnxSessionOptions() hardcoded
// intraOpNumThreads: 1 on every platform. That single-thread setting was the
// "conservative half" of a fix for a macOS crash (BFCArena::Extend /
// posix_memalign on 16GB Apple Silicon), but it shipped globally, so Whisper
// inference ran on one core on Windows and Linux too.
//
// Cost, from a packaged Windows build's own telemetry:
//
//   [LocalWhisperSTT/whisper-small:mic] latency · final: n=1 p50=12176ms
//
// 12.2s for one segment — about 10x slower than the speech being transcribed.
// Segments finish after the meeting ends, so the app appears to produce no
// transcript while nothing actually errors. That is the failure this pins.
//
// macOS intentionally stays at 1: the crash it mitigates is real and
// platform-specific. Elsewhere the count scales with the machine but stays
// bounded (half the cores, max 4) so one session cannot monopolise the box.
//
// Run: node --test 'electron/utils/__tests__/OnnxThreadDefaults.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODULE_URL = pathToFileURL(
  path.resolve(__dirname, '../../../dist-electron/electron/utils/onnxThreadConfig.js')
).href;

describe('ONNX session thread defaults', () => {
  test('non-macOS scales intra-op threads with cores instead of pinning to 1', async (t) => {
    if (process.platform === 'darwin') {
      t.skip('macOS deliberately stays single-threaded — see the crash note');
      return;
    }
    delete process.env.NATIVELY_ONNX_INTRA_OP_THREADS;
    const { getBoundedOnnxSessionOptions } = await import(MODULE_URL);
    const cores = os.cpus()?.length ?? 1;
    const opts = getBoundedOnnxSessionOptions();

    // Leave 2 logical processors for UI/audio/Electron, then SPLIT what remains
    // across the two concurrent Whisper sessions (mic + system audio) rather
    // than handing the full budget to each. Floor of 2 on any multi-core box.
    const budget = Math.max(1, cores - 2);
    const expected = Math.max(cores > 2 ? 2 : 1, Math.floor(budget / 2));
    assert.equal(
      opts.intraOpNumThreads,
      expected,
      `expected (${cores} cores - 2) split across 2 channels, min 2`,
    );

    // The point of the fix: on any multi-core machine this must exceed 1.
    if (cores > 1) {
      assert.ok(
        opts.intraOpNumThreads > 1,
        'multi-core machines must not run Whisper inference single-threaded',
      );
    }
  });

  test('two concurrent channels cannot oversubscribe the machine', async () => {
    // THE BUG: each channel took `cores - 2` for itself. On a 6-core/12-thread
    // i5-11260H that is 10 threads twice over — 20 threads on 12 logical
    // processors, with the OS preempting ONNX's spin-waits. Measured on that
    // machine, whisper-tiny.en (39MB) needed 2.9s for 1.89s of speech while
    // large-v3-turbo ran beside it. Tiny should beat real time many times over.
    //
    // Asserted against a synthetic 12-core machine so the check does not depend
    // on whatever the test runner happens to have.
    const { getBoundedOnnxSessionOptions } = await import(MODULE_URL);
    const realCpus = os.cpus;
    os.cpus = () => new Array(12).fill({ model: 'synthetic', speed: 0 });
    try {
      const perChannel = getBoundedOnnxSessionOptions().intraOpNumThreads;
      assert.ok(
        perChannel * 2 <= 12,
        `two channels at ${perChannel} threads each would oversubscribe a 12-thread CPU`,
      );
      assert.ok(perChannel >= 2, 'a channel must still get real parallelism');
    } finally {
      os.cpus = realCpus;
    }
  });

  test('always leaves headroom — never claims every logical processor', async () => {
    const { getBoundedOnnxSessionOptions } = await import(MODULE_URL);
    const cores = os.cpus()?.length ?? 1;
    const opts = getBoundedOnnxSessionOptions();
    assert.ok(opts.intraOpNumThreads >= 1, 'must always be at least 1');
    if (cores > 2) {
      assert.ok(
        opts.intraOpNumThreads < cores,
        'UI, audio capture and the rest of Electron still need a thread to run on',
      );
    }
  });

  test('env override still wins, so a bad machine can be pinned back to 1', async () => {
    process.env.NATIVELY_ONNX_INTRA_OP_THREADS = '1';
    try {
      const { getBoundedOnnxSessionOptions } = await import(MODULE_URL);
      assert.equal(getBoundedOnnxSessionOptions().intraOpNumThreads, 1);
    } finally {
      delete process.env.NATIVELY_ONNX_INTRA_OP_THREADS;
    }
  });

  test('a fresh options object is returned per call (transformers.js mutates it)', async () => {
    const { getBoundedOnnxSessionOptions } = await import(MODULE_URL);
    const a = getBoundedOnnxSessionOptions();
    const b = getBoundedOnnxSessionOptions();
    assert.notEqual(a, b, 'session_options must not be shared across sessions');
    assert.deepEqual(a, b);
  });
});
