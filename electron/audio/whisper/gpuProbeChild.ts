/**
 * DirectML probe — runs in a THROWAWAY CHILD PROCESS.
 *
 * Nothing in this file may be imported by the app. It exists to be executed as
 * a standalone script under `ELECTRON_RUN_AS_NODE=1`, so that when DirectML
 * initialisation aborts — and it aborts natively, without throwing anything JS
 * can catch — the casualty is this process and not Natively.
 *
 * That is the whole point. Two earlier GPU attempts put the DirectML session
 * creation inside the app's own worker thread; a native abort there killed the
 * app at launch, which is far worse than transcribing slowly. Here the parent
 * simply observes an exit code.
 *
 * Contract:
 *   argv[2] = absolute path to a real .onnx file to open
 *   argv[3] = DirectML adapter index (deviceId), or 'cpu' for the control run
 *   stdout  = one line of JSON on success: {"ok":true,"ms":1234}
 *   exit 0  = this adapter can create a DirectML session
 *   exit !0 = it cannot (or it took the process down trying)
 *
 * The 'cpu' mode is the control. If opening this model dies on the CPU too then
 * the model is the problem and DirectML is innocent — a distinction worth
 * having, because "GPU unavailable" and "this checkpoint is broken" produce the
 * same silent fall back to slow transcription.
 */

const onnxPath = process.argv[2];
const rawDevice = process.argv[3] ?? '0';
const cpuOnly = rawDevice.toLowerCase() === 'cpu';
const deviceId = cpuOnly ? -1 : Number.parseInt(rawDevice, 10);

async function main(): Promise<void> {
  if (!onnxPath) {
    process.stdout.write(JSON.stringify({ ok: false, error: 'no model path' }) + '\n');
    process.exit(2);
  }

  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const ort = require('onnxruntime-node');

  const t0 = Date.now();
  // Both flags are mandatory for DirectML — see getDirectMLSessionOptions().
  // Getting them wrong is what made the previous attempts fatal, so the probe
  // must use exactly the options the real session will use, or it proves
  // nothing about the real session.
  const session = await ort.InferenceSession.create(onnxPath, {
    executionProviders: cpuOnly ? ['cpu'] : [{ name: 'dml', deviceId }, 'cpu'],
    executionMode: 'sequential',
    enableMemPattern: false,
    enableCpuMemArena: false,
    graphOptimizationLevel: 'all',
  });

  const ms = Date.now() - t0;
  // Releasing explicitly keeps the GPU allocation from outliving the probe on
  // drivers that are slow to reclaim on exit.
  try { await session.release?.(); } catch { /* best-effort */ }

  process.stdout.write(JSON.stringify({ ok: true, ms, deviceId }) + '\n');
  process.exit(0);
}

main().catch((e: any) => {
  process.stdout.write(
    JSON.stringify({ ok: false, deviceId, error: String(e?.message ?? e).slice(0, 400) }) + '\n',
  );
  process.exit(1);
});
