// electron/audio/__tests__/ElectronBuilderFastConfig.test.mjs
//
// THE BUG THIS PINS: the CI packaging step was given `-c.compression=store` on
// the command line. electron-builder documents that dot-notation, but GitHub's
// windows-latest runners run `run:` steps under PowerShell, which does not pass
// it through intact. electron-builder read `-c` as --config and
// `.compression=store` as the config FILENAME, and the whole build died two
// seconds in:
//
//   ⨯ ENOENT: no such file or directory, open '...\.compression=store'
//       at readConfig (app-builder-lib/src/util/config/load.ts:19:16)
//
// The setting now lives in electron-builder.fast.cjs, which has no quoting or
// shell-parsing hazard. That file must keep inheriting package.json `build`
// wholesale — a hand-copied config would silently drift from the real one and
// produce an installer missing asarUnpack entries or NSIS settings.
//
// Run: node --test 'electron/audio/__tests__/ElectronBuilderFastConfig.test.mjs'

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '../../..');
const require_ = createRequire(import.meta.url);

const fastConfig = require_(path.join(repoRoot, 'electron-builder.fast.cjs'));
const base = require_(path.join(repoRoot, 'package.json')).build;
const workflow = fs.readFileSync(
  path.join(repoRoot, '.github/workflows/build-app.yml'),
  'utf8',
);
// The comments in that file quote the broken flag and the error it produced, so
// they must be stripped before asserting on what the workflow actually RUNS.
const workflowCommands = workflow
  .split('\n')
  .filter((line) => !/^\s*#/.test(line))
  .join('\n');

describe('CI packaging config', () => {
  test('compression is store — the point of the file', () => {
    assert.equal(fastConfig.compression, 'store');
  });

  test('everything else is inherited, not copied', () => {
    // Identity comparison, so a duplicated-then-edited value fails here.
    const drifted = Object.keys(fastConfig)
      .filter((k) => k !== 'compression')
      .filter((k) => fastConfig[k] !== base[k]);
    assert.deepEqual(drifted, [], 'these keys diverged from package.json build');
  });

  test('every base key survives the spread', () => {
    for (const key of Object.keys(base)) {
      assert.ok(key in fastConfig, `${key} was dropped from the CI config`);
    }
  });

  test('the packed-file rules the app depends on came through', () => {
    // A packaging config that loses asarUnpack ships an app whose native addons
    // and worker scripts cannot load — an installer that builds and then fails
    // at runtime, which is worse than a build that fails.
    assert.ok(Array.isArray(fastConfig.asarUnpack));
    for (const entry of ['**/*.node', '**/whisperWorker.js', '**/gpuProbeChild.js']) {
      assert.ok(fastConfig.asarUnpack.includes(entry), `asarUnpack lost ${entry}`);
    }
    assert.equal(fastConfig.appId, base.appId, 'a changed appId would not upgrade in place');
  });

  test('package.json itself is NOT slowed down for real users', () => {
    // Setting compression there would apply to the signed macOS release too.
    assert.equal(
      base.compression,
      undefined,
      'compression belongs in the CI-only config, not the shared one',
    );
  });
});

describe('the workflow invokes it by file', () => {
  test('no -c dot-notation survives anywhere in the workflow', () => {
    assert.ok(
      !/-c\.\w/.test(workflowCommands),
      'PowerShell does not pass electron-builder dot-notation through',
    );
  });

  test('the packaging step points at the config file', () => {
    assert.match(
      workflowCommands,
      /electron-builder --win --x64 --publish never --config electron-builder\.fast\.cjs/,
    );
  });

  test('the config file the workflow names actually exists', () => {
    const named = workflowCommands.match(/--config (\S+)/)?.[1];
    assert.ok(named, 'no --config argument found');
    assert.ok(
      fs.existsSync(path.join(repoRoot, named)),
      `workflow references ${named}, which is not in the repo`,
    );
  });
});
