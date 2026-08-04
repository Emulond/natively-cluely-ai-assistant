// electron/utils/__tests__/OnnxSlotWhisperStarvation.test.mjs
//
// Regression coverage for the ONNX slot cap starving the microphone channel.
//
// THE BUG THIS PINS: canAcquireNow() admitted 'high' only while
// inFlightNormal + inFlightHigh < cap (default 2). Long-lived 'normal'
// consumers hold their slot for the process lifetime — IntentClassifier stashes
// its release on the instance after a successful load, as does LocalReranker.
// So during a meeting the occupancy was:
//
//   IntentClassifier   'normal'  held for the process lifetime
//   Whisper channel #1 'high'    system audio (start() runs first, takes the
//                                preloader's warm worker)
//   Whisper channel #2 'high'    microphone — BLOCKED, cap already full
//
// The microphone instance then sat in acquireOnnxSlot for the whole meeting.
// Every VAD segment it produced hit dispatchFinal's `if (!this.worker) return`
// and was discarded. Observable result: audio captured, level meter moving, no
// transcript, and no error anywhere. Confirmed in a production log where the
// mic worker's "Cold-starting worker" line landed 16ms AFTER the meeting
// stopped — i.e. the instant channel #1 released its slot, 45s too late.
//
// 'high' blocking new 'normal' acquisitions does not help here: it does not
// preempt an already-running normal session, which is exactly what a
// process-lifetime holder is.
//
// THE FIX: high-priority acquisitions get dedicated headroom
// (HIGH_PRIORITY_RESERVED_SLOTS = 2) covering the two audio channels, so both
// Whisper sessions are admitted regardless of long-lived normal holders.
//
// Run: node --test 'electron/utils/__tests__/OnnxSlotWhisperStarvation.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODULE_URL = pathToFileURL(
  path.resolve(__dirname, '../../../dist-electron/electron/utils/onnxThreadConfig.js')
).href;

/** Resolves true if p settles before `ms`, false if it is still pending. */
function settlesWithin(p, ms) {
  let timer;
  const timeout = new Promise(resolve => { timer = setTimeout(() => resolve(false), ms); });
  return Promise.race([p.then(() => true), timeout]).finally(() => clearTimeout(timer));
}

describe('ONNX slot gate — Whisper channels are never starved', () => {
  test('both Whisper channels acquire while a process-lifetime normal holder sits on a slot', async () => {
    const mod = await import(MODULE_URL);
    mod.__resetOnnxGateForTests();

    // IntentClassifier loads and holds its slot for the process lifetime.
    const intentClassifier = await mod.acquireOnnxSlot('normal');

    // System audio starts first and takes a high slot for the meeting.
    const systemAudio = await mod.acquireOnnxSlot('high');

    // The microphone channel must still be admitted. Before the fix this
    // promise stayed pending until systemAudio released at meeting end.
    const micAcquired = await settlesWithin(mod.acquireOnnxSlot('high'), 250);
    assert.equal(
      micAcquired,
      true,
      'microphone Whisper channel must acquire a slot while IntentClassifier holds one',
    );

    intentClassifier();
    systemAudio();
    mod.__resetOnnxGateForTests();
  });

  test('high-priority headroom is bounded — a third high acquisition still queues', async () => {
    const mod = await import(MODULE_URL);
    mod.__resetOnnxGateForTests();

    // The two audio channels consume the reserved headroom.
    const a = await mod.acquireOnnxSlot('high');
    const b = await mod.acquireOnnxSlot('high');

    // A third high session is beyond both the reserve and the shared cap, so
    // it must queue — the reserve is per audio channel, not unbounded.
    const third = mod.acquireOnnxSlot('high');
    assert.equal(
      await settlesWithin(third, 150),
      false,
      'a third concurrent high session must queue',
    );

    // Releasing one channel hands the slot to the queued waiter.
    a();
    const thirdRelease = await third;
    thirdRelease();
    b();
    mod.__resetOnnxGateForTests();
  });

  test('normal-priority acquisition still queues behind waiting high-priority callers', async () => {
    const mod = await import(MODULE_URL);
    mod.__resetOnnxGateForTests();

    // Both channels hold the reserve, which also reaches the shared cap of 2.
    const h1 = await mod.acquireOnnxSlot('high');
    const h2 = await mod.acquireOnnxSlot('high');

    // Queue a high waiter, then attempt a normal acquisition. The normal must
    // not jump the queue ahead of latency-critical Whisper.
    const queuedHigh = mod.acquireOnnxSlot('high');
    assert.equal(
      await settlesWithin(mod.acquireOnnxSlot('normal'), 150),
      false,
      'normal must not acquire while a high-priority caller is queued',
    );

    h1();
    (await queuedHigh)();
    h2();
    mod.__resetOnnxGateForTests();
  });
});
