// electron/audio/__tests__/LocalWhisperFinalTaskWatchdog.test.mjs
//
// THE BUG THIS PINS: only STREAMING tasks were watchdogged. Finals — the
// messages that actually carry the transcript — were dispatched with no timer
// at all, so a final that never came back pinned its occupancy slot forever.
// With MAX_INFLIGHT_TRANSCRIBES = 2, two wedged finals close the channel
// permanently. From a packaged Windows build running whisper-large-v3-turbo:
//
//   transcribe START task=t1 samples=25440 (1.59s)     <- never returned
//   pipeline · dispatched=1 inFlight=1
//   pipeline · dispatched=1 inFlight=2
//   dropping a final segment — 2 transcriptions already queued   x8
//   pipeline · ... inFlight=2 droppedBackpressure=2
//
// inFlight sat at 2 for the rest of the meeting and every subsequent phrase was
// refused. No error, no crash — the channel simply stopped producing text.
//
// The worker cannot detect this itself: its own STILL RUNNING watchdog never
// logged once across 65 seconds of inference, because heavy ONNX work starves
// that thread's timers. The host's loop stays free, so the host must hold the
// stopwatch.
//
// Run: node --test 'electron/audio/__tests__/LocalWhisperFinalTaskWatchdog.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(
  path.resolve(__dirname, '../LocalWhisperSTT.ts'),
  'utf8',
);

describe('final-transcribe watchdog', () => {
  test('every dispatched task is armed, not just streaming ones', () => {
    // sendTranscribe() is the single dispatch path for both kinds.
    assert.match(
      source,
      /private\s+sendTranscribe\([^)]*\)[^{]*\{[\s\S]{0,400}?this\.armTaskWatchdog\(/,
      'armTaskWatchdog must be called from sendTranscribe so finals are covered',
    );
  });

  test('an expired task gives its queue slot back', () => {
    // Without this the backpressure limit converts a slow model into a silent
    // channel — which is exactly what shipped.
    // The watchdog's own release, and the release path it calls into. The
    // window is generous because the callback now also declines to blame a slow
    // model when the session is simply shutting down.
    const armIdx = source.indexOf('private armTaskWatchdog(');
    const releaseIdx = source.indexOf('this.noteTranscribeResolved(taskId)', armIdx);
    const nextMethodIdx = source.indexOf('private shortModelName()', armIdx);
    assert.ok(releaseIdx > armIdx, 'the watchdog must release the slot');
    assert.ok(
      releaseIdx < nextMethodIdx,
      'and it must do so inside armTaskWatchdog, so later speech is still accepted',
    );
  });

  test('the slot is released exactly once per task', () => {
    // A task that times out and then answers anyway must not be counted down
    // twice — that under-counts occupancy and re-opens the unbounded queue.
    assert.match(source, /releasedTasks/);
    assert.match(
      source,
      /if\s*\(this\.releasedTasks\.has\(taskId\)\)\s*return/,
      'a second release for the same taskId must be a no-op',
    );
  });

  test('the release memory is bounded', () => {
    assert.match(source, /RELEASED_TASK_MEMORY/);
    assert.match(
      source,
      /this\.releasedTasks\.size\s*>\s*LocalWhisperSTT\.RELEASED_TASK_MEMORY/,
      'the released-task set must not grow for the life of the process',
    );
  });

  test('the budget scales with audio length and is capped', () => {
    // A 14s segment legitimately takes longer than a 1s one; past the cap the
    // result is too late to be worth anything.
    assert.match(source, /TASK_WATCHDOG_MIN_MS/);
    assert.match(source, /TASK_WATCHDOG_MAX_MS/);
    assert.match(
      source,
      /Math\.min\(\s*LocalWhisperSTT\.TASK_WATCHDOG_MAX_MS,[\s\S]{0,200}?audioSeconds/,
      'the timeout must be derived from the audio duration, not fixed',
    );
  });

  test('result, error and partial all resolve by taskId', () => {
    for (const call of [
      /msg\.type === 'partial'[\s\S]{0,400}?this\.noteTranscribeResolved\(msg\.taskId\)/,
      /msg\.type === 'result'[\s\S]{0,200}?this\.noteTranscribeResolved\(msg\.taskId\)/,
      /msg\.type === 'error'[\s\S]{0,200}?this\.noteTranscribeResolved\(msg\.taskId\)/,
    ]) {
      assert.match(source, call, 'resolution must clear the task-specific watchdog');
    }
  });

  test('a dead worker resets occupancy instead of leaking it', () => {
    // Both handlers must clear the watchdogs AND zero the occupancy counter.
    // Not asserted as adjacent lines: the held-audio buffer is now cleared
    // between them, and that ordering is not what matters.
    const resets = source.match(
      /this\.clearAllTaskWatchdogs\(\);[\s\S]{0,200}?this\.inFlightTranscribes = 0;/g,
    ) ?? [];
    assert.ok(
      resets.length >= 2,
      'both the worker error and worker exit handlers must clear outstanding tasks',
    );
  });

  test('the diag line names outstanding tasks', () => {
    // "inFlight=2" alone cannot distinguish a busy pipeline from a wedged one;
    // stable ids across consecutive lines are the signature of wedged.
    assert.match(source, /outstanding=\$\{\[\.\.\.this\.taskWatchdogs\.keys\(\)\]/);
  });

  test('a task outstanding at session end is not reported as a fault', () => {
    assert.match(
      source,
      /if\s*\(!this\.isActive\s*&&\s*!this\.isDrainingFinals\)\s*\{[\s\S]{0,200}?return;/,
      'shutting down mid-inference is expected, not a slow-model warning',
    );
  });
});
