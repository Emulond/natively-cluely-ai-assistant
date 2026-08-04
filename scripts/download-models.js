const path = require('path');
const fs = require('fs');

// Required core-fallback model files. The BGE reranker is also required for
// smart-retrieval Phase 1/3 (confidence-gated local rerank escalation) and is
// bundled so a clean-machine install never has to download a 280MB cross-encoder
// on first document-grounded mode activation.
const REQUIRED_MODEL_FILES = [
    'Xenova/all-MiniLM-L6-v2/config.json',
    'Xenova/all-MiniLM-L6-v2/tokenizer.json',
    'Xenova/all-MiniLM-L6-v2/tokenizer_config.json',
    'Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx',
    'Xenova/mobilebert-uncased-mnli/config.json',
    'Xenova/mobilebert-uncased-mnli/tokenizer.json',
    'Xenova/mobilebert-uncased-mnli/tokenizer_config.json',
    'Xenova/mobilebert-uncased-mnli/onnx/model_quantized.onnx',
    'Xenova/bge-reranker-base/config.json',
    'Xenova/bge-reranker-base/tokenizer.json',
    'Xenova/bge-reranker-base/tokenizer_config.json',
    'Xenova/bge-reranker-base/onnx/model_quantized.onnx',
];

function verifyModels() {
    const modelsDir = path.join(__dirname, '../resources/models');
    const missing = [];
    for (const rel of REQUIRED_MODEL_FILES) {
        const full = path.join(modelsDir, rel);
        let ok = false;
        try { ok = fs.existsSync(full) && fs.statSync(full).size > 0; } catch { ok = false; }
        if (!ok) missing.push(full);
    }
    if (missing.length > 0) {
        console.error('[download-models] VERIFY FAILED — required model files missing or empty:');
        for (const m of missing) console.error('  ✗', m);
        process.exit(1);
    }
    console.log('[download-models] VERIFY OK — all required core-fallback model files present.');
}

/**
 * Runs `fn`, retrying on failure.
 *
 * pipeline() both downloads a model AND opens an ONNX session on it, so this
 * step fails for two very different reasons. Observed twice on the Windows CI
 * runner, each time costing a full ~20 minute build:
 *
 *   [download-models] Error downloading model: Error: Load model from
 *     ...\bge-reranker-base\onnx\model_quantized.onnx failed:system error number 13
 *       at new OnnxruntimeSessionHandler (...onnxruntime-node\dist\backend.js:50:92)
 *
 * The bytes arrived; ORT could not open them. Error 13 on Windows is an access
 * failure — typically a just-written ~280MB file still held by the filesystem
 * or an antivirus scan. A short wait clears that, so plain retries handle it.
 *
 * A genuinely truncated download would not, which is why the FINAL attempt
 * deletes the cached model directory first and re-fetches from scratch. That
 * costs bandwidth, so it is the last resort rather than the first.
 */
async function withRetry(label, fn, modelDir) {
    const ATTEMPTS = 3;
    for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
        try {
            return await fn();
        } catch (e) {
            if (attempt === ATTEMPTS) {
                console.error(`[download-models] ${label} failed after ${ATTEMPTS} attempts.`);
                throw e;
            }
            console.warn(
                `[download-models] ${label} attempt ${attempt}/${ATTEMPTS} failed: ${e?.message ?? e}`,
            );
            // Last chance: assume the cached copy is bad and start clean.
            if (attempt === ATTEMPTS - 1 && modelDir && fs.existsSync(modelDir)) {
                console.warn(`[download-models] Removing possibly-corrupt cache before final attempt: ${modelDir}`);
                try {
                    fs.rmSync(modelDir, { recursive: true, force: true });
                } catch (rmErr) {
                    console.warn(`[download-models] Could not remove ${modelDir}: ${rmErr?.message ?? rmErr}`);
                }
            }
            const delayMs = 3000 * attempt;
            console.warn(`[download-models] Retrying ${label} in ${delayMs}ms...`);
            await new Promise((resolve) => setTimeout(resolve, delayMs));
        }
    }
}

async function downloadModels() {
    const { pipeline, env } = await import('@huggingface/transformers');
    const modelsDir = path.join(__dirname, '../resources/models');
    const modelDirFor = (repoId) => path.join(modelsDir, ...repoId.split('/'));
    
    // Ensure the directory exists
    if (!fs.existsSync(modelsDir)) {
        fs.mkdirSync(modelsDir, { recursive: true });
    }

    // Let Transformers.js handle the download but specify the local directory cache
    env.cacheDir = modelsDir;
    
    try {
        // 1. Embedding model (RAG)
        console.log('[download-models] Downloading Xenova/all-MiniLM-L6-v2...');
        await withRetry(
            'all-MiniLM-L6-v2',
            () => pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2'),
            modelDirFor('Xenova/all-MiniLM-L6-v2'),
        );
        console.log('[download-models] all-MiniLM-L6-v2 downloaded.');

        // 2. Zero-shot classification model (Intent Classifier)
        console.log('[download-models] Downloading Xenova/mobilebert-uncased-mnli...');
        await withRetry(
            'mobilebert-uncased-mnli',
            () => pipeline('zero-shot-classification', 'Xenova/mobilebert-uncased-mnli'),
            modelDirFor('Xenova/mobilebert-uncased-mnli'),
        );
        console.log('[download-models] mobilebert-uncased-mnli downloaded.');

        // 3. Cross-encoder reranker (smart-retrieval Phase 1/3 — confidence-gated
        //    rerank escalation). Bundled in resources/models/ so a clean-machine
        //    install can do offline rerank without a 280MB first-activation
        //    download. The installer ships the q8 quantized variant (~280MB).
        //
        //    The lazy-download provider in electron/rag/rerankerDownloadProvider.ts
        //    still acts as a no-op fallback if the bundled model is absent
        //    (e.g. an old installer predating this bundling).
        console.log('[download-models] Downloading Xenova/bge-reranker-base (q8)...');
        // Use dtype:'q8' so transformers.js selects the quantized ONNX variant
        // (~280 MB) instead of the fp32 one (~1.1 GB). NATIVELY_RERANKER_DTYPE
        // override remains for accuracy experiments.
        const rerankerDtype = (process.env.NATIVELY_RERANKER_DTYPE || 'q8').trim() || 'q8';
        await withRetry(
            'bge-reranker-base',
            () => pipeline('text-classification', 'Xenova/bge-reranker-base', { dtype: rerankerDtype }),
            modelDirFor('Xenova/bge-reranker-base'),
        );
        console.log('[download-models] bge-reranker-base downloaded.');

        console.log('[download-models] All models downloaded successfully!');
    } catch (e) {
        console.error('[download-models] Error downloading model:', e);
        process.exit(1);
    }
}

if (process.argv.includes('--verify')) {
    // Fail-loud, no-network check that required models are already on disk.
    verifyModels();
} else {
    downloadModels().catch((e) => {
        console.error('[download-models] Fatal error:', e);
        process.exit(1);
    });
}

