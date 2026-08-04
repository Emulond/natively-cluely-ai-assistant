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

    // Leave 2 threads for UI/audio/Electron, take the rest, capped at 8.
    const expected = Math.max(1, cores - 2);
    assert.equal(
      opts.intraOpNumThreads,
      expected,
      `expected ${cores} cores minus 2, capped at 8`,
    );

    // The point of the fix: on any multi-core machine this must exceed 1.
    if (cores > 1) {
      assert.ok(
        opts.intraOpNumThreads > 1,
        'multi-core machines must not run Whisper inference single-threaded',
      );
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
