/**
 * Deciding whether DirectML is safe to use — without betting the app on it.
 *
 * BACKGROUND. Two earlier attempts at GPU inference crashed Natively at launch.
 * Both put the DirectML session creation inside the app's own process: the
 * first passed `device: 'dml'` to transformers.js, the second built an
 * automatic adapter ladder. DirectML failures surface as a NATIVE abort, so
 * there is no exception to catch and no chance to fall back — the process is
 * simply gone. A sentinel file written just before the attempt could not save
 * it either, because the abort can beat the write to disk. The result was an
 * app that would not open at all, which is far worse than one that transcribes
 * slowly.
 *
 * THE FIX IS STRUCTURAL, NOT DEFENSIVE. The attempt happens in a throwaway
 * child process (gpuProbeChild.ts). If DirectML aborts, the child dies and the
 * parent reads an exit code — an observable event instead of a fatal one. Only
 * an adapter that has survived a real session creation is ever handed to the
 * transcription worker.
 *
 * WHY IT WAS FAILING. ONNX Runtime documents two hard requirements for the
 * DirectML provider: memory-pattern optimisation must be off and execution must
 * be sequential, "or an error will be returned". The shared session options
 * enable memory patterns (they help CPU throughput), and those options were
 * being passed straight into the DirectML session. Every attempt was therefore
 * guaranteed to fail before it began. See getDirectMLSessionOptions().
 *
 * WHICH ADAPTER. Laptops have two, and the useful one is the discrete GPU. The
 * integrated GPU shares system RAM and is often slower than the CPU path it
 * would replace. Electron's own GPU enumeration gives vendor ids, so the
 * discrete adapter can be identified by vendor (NVIDIA 0x10DE, AMD 0x1002)
 * rather than guessed — and every candidate is probed in order, so a wrong
 * guess costs a child process rather than a session.
 *
 * COST. One child process per adapter candidate, once per (GPU, driver, ORT
 * version). The verdict is cached in userData and only re-probed when one of
 * those changes — a driver update is exactly when a previously broken adapter
 * might start working, and a previously working one might stop.
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { resolveGpuProbeChildPath } from './workerPathResolver';

export interface GpuProbeResult {
    /** DirectML adapter index proven to create a session, or null for CPU. */
    deviceId: number | null;
    /** Why — carried into the log so the choice is never a mystery. */
    reason: string;
    /** Cache key components, so a driver or runtime change re-probes. */
    signature: string;
    probedAt: number;
    /** Per-adapter outcomes, for diagnostics. */
    attempts: Array<{ deviceId: number; ok: boolean; ms?: number; error?: string }>;
}

const PROBE_TIMEOUT_MS = 45_000;
const CACHE_FILENAME = 'gpu-probe.json';

const VENDOR_NVIDIA = 0x10de;
const VENDOR_AMD = 0x1002;
const VENDOR_INTEL = 0x8086;

function vendorName(id: number | undefined): string {
    switch (id) {
        case VENDOR_NVIDIA: return 'NVIDIA';
        case VENDOR_AMD: return 'AMD';
        case VENDOR_INTEL: return 'Intel';
        default: return `vendor 0x${(id ?? 0).toString(16)}`;
    }
}

/**
 * Adapter indices worth probing, discrete first.
 *
 * Electron reports the adapters Chromium enumerated, in DXGI order on Windows —
 * the same order DirectML's deviceId indexes into. Ordering by "discrete
 * vendor first" means an NVIDIA or AMD card is tried before the integrated
 * Intel one even when the integrated one is adapter 0, which it usually is.
 *
 * A machine whose enumeration cannot be read still gets [0], the ORT default —
 * probed like any other candidate, so an unknown machine degrades to one dead
 * child process rather than a crash.
 */
export async function orderedAdapterCandidates(): Promise<Array<{ deviceId: number; vendor: string }>> {
    try {
        // Required lazily: this module is imported by code that runs before the
        // Electron app is ready in some paths.
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { app } = require('electron');
        const info: any = await app.getGPUInfo('complete');
        const devices: any[] = Array.isArray(info?.gpuDevice) ? info.gpuDevice : [];
        if (devices.length === 0) return [{ deviceId: 0, vendor: 'unknown' }];

        const candidates = devices.map((d, index) => ({
            deviceId: index,
            vendor: vendorName(d?.vendorId),
            discrete: d?.vendorId === VENDOR_NVIDIA || d?.vendorId === VENDOR_AMD,
        }));
        candidates.sort((a, b) => Number(b.discrete) - Number(a.discrete));
        return candidates.map(({ deviceId, vendor }) => ({ deviceId, vendor }));
    } catch {
        return [{ deviceId: 0, vendor: 'unknown' }];
    }
}

/**
 * A real .onnx file to open. Any will do — what is being tested is whether the
 * DirectML device initialises at all, which is a device-level property, so the
 * smallest cached model is used to keep the probe quick.
 *
 * Returns null when no model has been downloaded yet. There is nothing to
 * accelerate in that state, and inventing a model to probe with would mean
 * shipping a binary blob for no benefit.
 */
export function findSmallestCachedOnnx(modelsDir: string): string | null {
    let best: { file: string; size: number } | null = null;
    const walk = (dir: string, depth: number): void => {
        if (depth > 6) return;
        let entries: fs.Dirent[];
        try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
        for (const entry of entries) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) { walk(full, depth + 1); continue; }
            if (!entry.name.endsWith('.onnx')) continue;
            try {
                const { size } = fs.statSync(full);
                // Skip external-data stubs: a few hundred KB of graph whose
                // weights live in a sibling .onnx_data. Opening one without its
                // companion fails for reasons that have nothing to do with the
                // GPU, which would poison the verdict.
                if (size < 1_000_000) continue;
                if (!best || size < best.size) best = { file: full, size };
            } catch { /* unreadable — skip */ }
        }
    };
    walk(modelsDir, 0);
    return best ? (best as { file: string }).file : null;
}

function runProbeChild(onnxPath: string, deviceId: number): Promise<{ ok: boolean; ms?: number; error?: string }> {
    return new Promise((resolve) => {
        const script = resolveGpuProbeChildPath();
        if (!fs.existsSync(script)) {
            resolve({ ok: false, error: `probe script missing at ${script}` });
            return;
        }
        const child = spawn(process.execPath, [script, onnxPath, String(deviceId)], {
            env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
            stdio: ['ignore', 'pipe', 'pipe'],
            windowsHide: true,
        });

        let stdout = '';
        let stderr = '';
        let settled = false;
        const finish = (r: { ok: boolean; ms?: number; error?: string }) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            resolve(r);
        };

        // A hung probe is as useless as a failed one, and DirectML init can hang
        // on a wedged driver rather than aborting.
        const timer = setTimeout(() => {
            try { child.kill('SIGKILL'); } catch { /* already gone */ }
            finish({ ok: false, error: `timed out after ${PROBE_TIMEOUT_MS}ms` });
        }, PROBE_TIMEOUT_MS);

        child.stdout?.on('data', (d) => { stdout += String(d); });
        child.stderr?.on('data', (d) => { stderr += String(d); });
        child.on('error', (e) => finish({ ok: false, error: `spawn failed: ${e.message}` }));
        child.on('exit', (code, signal) => {
            if (code === 0) {
                // Exit 0 is NOT proof the GPU was used. ONNX Runtime quietly
                // drops a provider it cannot load and carries on:
                //
                //   removing requested execution provider "dml" from session
                //   options because it is not available: backend not found.
                //
                // The session then builds perfectly — on the CPU. Believing that
                // would mean reporting GPU acceleration on a machine that has
                // none, and the 'cpu' fallback we append to the provider list
                // guarantees it always succeeds. Only the absence of that
                // message means DirectML actually took the work.
                const combined = `${stdout}\n${stderr}`;
                if (/removing requested execution provider "?dml"?/i.test(combined)) {
                    finish({ ok: false, error: 'DirectML backend not available — ORT fell back to CPU' });
                    return;
                }
                let ms: number | undefined;
                try { ms = JSON.parse(stdout.trim().split('\n').pop() ?? '{}').ms; } catch { /* optional */ }
                finish({ ok: true, ms });
                return;
            }
            // This is the case the whole design exists for: a native abort took
            // the child down. In-process it would have taken the app down.
            const detail = signal
                ? `killed by ${signal}`
                : (stdout.trim() || stderr.trim() || `exit code ${code}`).slice(0, 300);
            finish({ ok: false, error: detail });
        });
    });
}

function cachePath(userDataDir: string): string {
    return path.join(userDataDir, CACHE_FILENAME);
}

/** GPU identity + runtime version. A change in any of these invalidates the verdict. */
async function buildSignature(): Promise<string> {
    let gpu = 'unknown';
    try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { app } = require('electron');
        const info: any = await app.getGPUInfo('complete');
        const devices: any[] = Array.isArray(info?.gpuDevice) ? info.gpuDevice : [];
        gpu = devices
            .map((d) => `${d?.vendorId}:${d?.deviceId}:${d?.driverVersion ?? ''}`)
            .join('|') || 'none';
    } catch { /* keep 'unknown' */ }
    let ortVersion = 'unknown';
    try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        ortVersion = require('onnxruntime-node/package.json').version ?? 'unknown';
    } catch { /* keep 'unknown' */ }
    return `${process.platform}/${process.arch}/ort${ortVersion}/${gpu}`;
}

export function readCachedProbe(userDataDir: string, signature: string): GpuProbeResult | null {
    try {
        const raw = fs.readFileSync(cachePath(userDataDir), 'utf8');
        const parsed = JSON.parse(raw) as GpuProbeResult;
        if (parsed?.signature !== signature) return null;
        return parsed;
    } catch {
        return null;
    }
}

function writeCachedProbe(userDataDir: string, result: GpuProbeResult): void {
    try {
        fs.writeFileSync(cachePath(userDataDir), JSON.stringify(result, null, 2));
    } catch { /* best-effort — a lost cache costs one re-probe */ }
}

/**
 * Resolve which DirectML adapter, if any, Whisper should use.
 *
 * Never throws and never crashes the caller: every failure path resolves to
 * `deviceId: null`, which means CPU.
 */
export async function resolveGpuDevice(opts: {
    userDataDir: string;
    modelsDir: string;
    /** Set NATIVELY_ONNX_DEVICE=cpu to opt out; =dml to force past a cached refusal. */
    force?: boolean;
}): Promise<GpuProbeResult> {
    const cpu = (reason: string): GpuProbeResult => ({
        deviceId: null, reason, signature: '', probedAt: Date.now(), attempts: [],
    });

    if (process.platform !== 'win32') {
        return cpu('DirectML is Windows-only; other platforms use their own provider selection');
    }
    if ((process.env.NATIVELY_ONNX_DEVICE ?? '').trim().toLowerCase() === 'cpu') {
        return cpu('disabled by NATIVELY_ONNX_DEVICE=cpu');
    }

    const signature = await buildSignature();

    if (!opts.force) {
        const cached = readCachedProbe(opts.userDataDir, signature);
        if (cached) {
            return {
                ...cached,
                reason: `${cached.reason} (cached; re-probes when GPU, driver or ONNX Runtime changes)`,
            };
        }
    }

    const onnxPath = findSmallestCachedOnnx(opts.modelsDir);
    if (!onnxPath) {
        // Not cached: with no model downloaded there is nothing to accelerate,
        // and the answer may differ once one exists.
        return cpu('no downloaded model to probe with — staying on CPU until one exists');
    }

    const candidates = await orderedAdapterCandidates();
    const attempts: GpuProbeResult['attempts'] = [];

    for (const { deviceId, vendor } of candidates) {
        const outcome = await runProbeChild(onnxPath, deviceId);
        attempts.push({ deviceId, ok: outcome.ok, ms: outcome.ms, error: outcome.error });
        if (outcome.ok) {
            const result: GpuProbeResult = {
                deviceId,
                reason: `DirectML adapter ${deviceId} (${vendor}) created a session in ${outcome.ms ?? '?'}ms`,
                signature,
                probedAt: Date.now(),
                attempts,
            };
            writeCachedProbe(opts.userDataDir, result);
            return result;
        }
    }

    const result: GpuProbeResult = {
        deviceId: null,
        reason: `no DirectML adapter could create a session (${attempts.length} tried) — using CPU`,
        signature,
        probedAt: Date.now(),
        attempts,
    };
    writeCachedProbe(opts.userDataDir, result);
    return result;
}

// ── Process-wide memo ───────────────────────────────────────────────────────
//
// buildWorkerInitMessage() is synchronous and runs on the path that spawns a
// transcription worker, so it cannot await a probe. The app kicks the probe off
// at startup instead — long before the first meeting — and the worker reads
// whatever has settled by then.
//
// A worker that starts before the probe finishes runs on the CPU. That is the
// correct trade: it is the outcome we already ship, arrived at without ever
// blocking meeting start on a GPU question.

let memoisedProbe: GpuProbeResult | null = null;
let probeInFlight: Promise<GpuProbeResult> | null = null;

/** The settled verdict, or null if the probe has not finished (⇒ use CPU). */
export function getResolvedGpuDevice(): GpuProbeResult | null {
    return memoisedProbe;
}

/**
 * Start (or join) the probe. Safe to call repeatedly; runs at most once per
 * process. Never rejects.
 */
export function ensureGpuProbe(opts: { userDataDir: string; modelsDir: string }): Promise<GpuProbeResult> {
    if (memoisedProbe) return Promise.resolve(memoisedProbe);
    if (probeInFlight) return probeInFlight;
    probeInFlight = resolveGpuDevice(opts)
        .catch((e) => ({
            deviceId: null,
            reason: `GPU probe failed outright (${String(e?.message ?? e).slice(0, 200)}) — using CPU`,
            signature: '',
            probedAt: Date.now(),
            attempts: [],
        } as GpuProbeResult))
        .then((result) => {
            memoisedProbe = result;
            probeInFlight = null;
            const detail = result.attempts.length
                ? ` attempts=${JSON.stringify(result.attempts)}`
                : '';
            console.log(`[GpuProbe] ${result.reason}${detail}`);
            return result;
        });
    return probeInFlight;
}

/** Test seam — lets a test install a verdict without spawning children. */
export function __setResolvedGpuDeviceForTests(result: GpuProbeResult | null): void {
    memoisedProbe = result;
    probeInFlight = null;
}
