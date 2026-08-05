/**
 * electron-builder.fast.cjs — CI config for the unsigned Windows workflow.
 *
 * Used ONLY by .github/workflows/build-app.yml via
 * `electron-builder --config electron-builder.fast.cjs`.
 *
 * WHAT IT CHANGES: `compression: 'store'`, and nothing else. Everything is
 * inherited from `package.json` `build`, the same way electron-builder.signed.cjs
 * inherits it, so the two configs cannot drift apart in anything but this one
 * field.
 *
 * WHY: packaging was 11m38s of an 18m30s run — 63% of the whole build — spent
 * LZMA-compressing an app that is mostly already-compressed ONNX model weights
 * and an Electron runtime. `store` skips that. The installer gets larger; the
 * build gets several times shorter, and this artifact is downloaded once from a
 * workflow run rather than served to users over a CDN.
 *
 * WHY NOT `-c.compression=store` ON THE COMMAND LINE: electron-builder does
 * document that dot-notation, but under PowerShell — which is what GitHub's
 * windows-latest runners use for `run:` steps — the argument does not survive
 * tokenisation. `-c` was taken as `--config` and `.compression=store` as its
 * value, so electron-builder went looking for a config file by that name:
 *
 *   ⨯ ENOENT: no such file or directory, open '...\.compression=store'
 *       at readConfig (app-builder-lib/src/util/config/load.ts:19:16)
 *
 * A real config file has no quoting or shell-parsing hazard at all.
 *
 * DO NOT set compression in package.json `build` instead — that would slow-path
 * every other build (including the signed macOS release) into shipping
 * uncompressed artifacts to real users.
 */

const base = require('./package.json').build;

module.exports = {
  ...base,
  compression: 'store',
};
