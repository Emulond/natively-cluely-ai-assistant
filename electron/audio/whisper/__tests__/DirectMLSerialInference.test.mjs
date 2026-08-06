// electron/audio/whisper/__tests__/DirectMLSerialInference.test.mjs
//
// THE BUG THIS PINS: several inferences ran concurrently on one DirectML
// session, which ONNX Runtime forbids —
//
//   "As the DirectML execution provider does not support parallel execution, it
//    does not support multi-threaded calls to Run on the same inference session.
//    That is, if an inference session using the DirectML execution provider,
//    only one thread may call Run at a time."
//   — onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
//
// The worker's message handler is `async`, so every `transcribe` message begins
// its own execution and the `await pipe(...)` calls interleave. On the CPU that
// was invisible: ORT's CPU path blocks the thread, so they serialised by
// accident. DirectML hands work to the GPU and returns to the event loop, which
// lets the next message straight in — and then nothing returns:
//
//   transcribe STILL RUNNING task=s1 elapsed=37832ms audio=0.90s
//   transcribe STILL RUNNING task=t2 elapsed=24702ms audio=2.49s
//   transcribe STILL RUNNING task=t3 elapsed=24508ms audio=2.07s
//   transcribe STILL RUNNING task=t4 elapsed=8757ms  audio=4.56s
//
// Four runs, none finishing, with real GPU utilisation the whole time. A
// deadlock with the fans on.
//
// Run: node --test 'electron/audio/whisper/__tests__/DirectMLSerialInference.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(
  path.resolve(__dirname, '../whisperWorker.ts'),
  'utf8',
);

describe('inference is serialised inside the worker', () => {
  test('a lock exists and the transcribe branch takes it', () => {
    assert.match(src, /function acquireInferenceLock\(\): Promise<\(\) => void>/);
    assert.match(
      src,
      /const releaseInference = await acquireInferenceLock\(\);\s*\n\s*try \{/,
      'the lock must be held across the whole transcribe body',
    );
  });

  test('it is taken BEFORE the model is invoked', () => {
    const acquireIdx = src.indexOf('await acquireInferenceLock()');
    const pipeIdx = src.indexOf('result = await pipe(msg.audio, opts)', acquireIdx);
    assert.ok(acquireIdx > 0, 'the lock must be acquired');
    assert.ok(pipeIdx > acquireIdx, 'and acquired before pipe() is called');
  });

  test('it is released however the branch ends', () => {
    // A failure that keeps the lock would wedge the channel permanently —
    // strictly worse than the bug being fixed.
    assert.match(
      src,
      /\} finally \{\s*\n[^}]*?releaseInference\(\);\s*\n\s*\}/,
      'release must be in a finally, not only on the success path',
    );
  });

  test('the queue chains rather than dropping waiters', () => {
    // Each waiter must resume the NEXT one, so N queued transcribes all run.
    assert.match(src, /inferenceLock = inferenceLock\.then\(\(\) => next, \(\) => next\)/);
    assert.match(
      src,
      /return waitFor\.then\(\(\) => release, \(\) => release\)/,
      'a rejected predecessor must still hand the lock on',
    );
  });

  test('the constraint is recorded, not just the workaround', () => {
    assert.match(src, /does not support parallel execution/);
    assert.match(src, /only one thread may call Run at a time/);
  });
});
