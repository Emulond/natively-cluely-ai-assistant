// Regression test: local Whisper must honor the user's recognition-language
// selection. The worker used to map the language through a BCP-47-keyed table
// ('ru-RU', 'en-US'), but the host sends the shared RECOGNITION_LANGUAGES key
// ('russian', 'english-us', 'auto') — the same value every other STT provider
// receives. Every selection missed → the language fell to null, and
// transformers.js 3.8.1 (which has no Whisper language auto-detection — see
// models.js `_retrieve_init_tokens`, it hardcodes English) forced English on
// multilingual checkpoints. Selecting Russian transcribed Russian speech as
// English.
//
// The worker contract is now: explicit shared keys resolve to their iso639 code
// ('russian' → 'ru'), 'auto' and unknown keys resolve to null (language left
// unforced), and English-only checkpoints still force 'english'.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { RECOGNITION_LANGUAGES } from '../../../dist-electron/electron/config/languages.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '../../..');

function readSrc(relPath) {
  return fs.readFileSync(path.resolve(root, relPath), 'utf8');
}

// Mirror of the worker's resolver, driven by the exact config the worker uses.
function resolveWhisperLanguage(key) {
  if (!key || key === 'auto') return null;
  return RECOGNITION_LANGUAGES[key]?.iso639 ?? null;
}

function stripComments(source) {
  // Remove line + block comments so string/content assertions only see code.
  let out = '';
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    const nx = source[i + 1];
    if (ch === '/' && nx === '/') {
      const nl = source.indexOf('\n', i);
      i = nl === -1 ? source.length : nl;
      continue;
    }
    if (ch === '/' && nx === '*') {
      const end = source.indexOf('*/', i + 2);
      i = end === -1 ? source.length : end + 2;
      continue;
    }
    out += ch;
    i++;
  }
  return out;
}

test('whisperWorker no longer uses a BCP-47-keyed LANG_MAP', () => {
  const code = stripComments(readSrc('electron/audio/whisper/whisperWorker.ts'));
  assert.ok(
    !code.includes('const LANG_MAP'),
    'whisperWorker must no longer hold a BCP-47-keyed LANG_MAP',
  );
  assert.ok(
    !code.includes("'ru-RU'") && !code.includes("'en-US'"),
    'whisperWorker must not key languages by BCP-47 tags',
  );
  assert.ok(
    code.includes('RECOGNITION_LANGUAGES[key]?.iso639'),
    'whisperWorker must resolve languages through RECOGNITION_LANGUAGES[key]?.iso639',
  );
  assert.ok(
    code.includes("import { RECOGNITION_LANGUAGES } from '../../config/languages'"),
    'whisperWorker must import the shared RECOGNITION_LANGUAGES config',
  );
  assert.ok(
    code.includes('resolveWhisperLanguage(msg.language)'),
    'transcribe handler must route the host language key through the resolver',
  );
});

test('explicit shared keys resolve to Whisper ISO-639-1 codes', () => {
  assert.equal(resolveWhisperLanguage('russian'), 'ru');
  assert.equal(resolveWhisperLanguage('english-us'), 'en');
  assert.equal(resolveWhisperLanguage('french'), 'fr');
  assert.equal(resolveWhisperLanguage('ukrainian'), 'uk');
  assert.equal(resolveWhisperLanguage('japanese'), 'ja');
});

test("'auto' and unknown keys resolve to null (language left unforced)", () => {
  assert.equal(resolveWhisperLanguage('auto'), null);
  assert.equal(resolveWhisperLanguage(''), null);
  assert.equal(resolveWhisperLanguage(undefined), null);
  assert.equal(resolveWhisperLanguage('ru-RU'), null, 'BCP-47 tags must NOT be accepted');
  assert.equal(resolveWhisperLanguage('klingon'), null);
});

test('every configured non-auto language declares a usable iso639 code', () => {
  for (const [key, lang] of Object.entries(RECOGNITION_LANGUAGES)) {
    if (key === 'auto') continue;
    assert.ok(lang.iso639, `language ${key} must declare iso639`);
    assert.match(
      lang.iso639,
      /^[a-z]{2}$/,
      `language ${key} iso639 '${lang.iso639}' must be a two-letter ISO-639-1 code`,
    );
  }
});

test('whisperWorker still forces english for English-only checkpoints', () => {
  const src = readSrc('electron/audio/whisper/whisperWorker.ts');
  assert.ok(
    src.includes('ENGLISH_ONLY_MODELS.has(loadedModelId)'),
    'English-only checkpoints must still force language=english',
  );
});
